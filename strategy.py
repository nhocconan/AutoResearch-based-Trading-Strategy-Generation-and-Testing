#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d ADX regime filter and volume spike confirmation
# - Primary: 6h Williams %R(14) for overbought/oversold conditions
# - HTF: 1d ADX > 20 to filter ranging markets + 1d volume > 2.0x 20-period MA for momentum confirmation
# - Long: Williams %R < -80 (oversold) + ADX > 20 + volume spike
# - Short: Williams %R > -20 (overbought) + ADX > 20 + volume spike
# - Exit: Williams %R returns to -50 level (mean reversion midpoint)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Williams %R captures short-term exhaustion, ADX filters chop, volume confirms conviction
# - Target: 60-120 trades over 4 years (15-30/year) to stay within fee drag limits

name = "6h_1d_williamsr_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for Williams %R and ADX
        return np.zeros(n)
    
    # Pre-compute 6h data
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams %R (14-period) on 6h
    period_wr = 14
    highest_high = pd.Series(high_6h).rolling(window=period_wr, min_periods=period_wr).max().values
    lowest_low = pd.Series(low_6h).rolling(window=period_wr, min_periods=period_wr).min().values
    williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low)
    
    # Calculate ADX (1d) for trend strength
    period_adx = 14
    # True Range
    tr1 = np.abs(np.roll(high_1d, 1) - np.roll(low_1d, 1))
    tr2 = np.abs(np.roll(high_1d, 1) - np.roll(close_1d, 1))
    tr3 = np.abs(np.roll(low_1d, 1) - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.roll(high_1d, 1) - high_1d
    down_move = low_1d - np.roll(low_1d, 1)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values (using Wilder's smoothing = EMA with alpha=1/period)
    alpha = 1.0 / period_adx
    
    atr = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().values / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().values
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume (aligned to 6h)
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        
        # Volume confirmation: current 1d volume > 2.0x 20-period MA
        volume_confirm = volume_1d_aligned[i] > 2.0 * volume_ma_20_1d_aligned[i]
        
        # ADX trend filter: ADX > 20 indicates non-ranging market
        trend_confirm = adx_aligned[i] > 20.0
        
        # Williams %R conditions
        oversold = williams_r_aligned[i] < -80.0
        overbought = williams_r_aligned[i] > -20.0
        exit_level = williams_r_aligned[i] > -50.0  # Return to midpoint
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Oversold + volume confirmation + trend confirmation
            if oversold and volume_confirm and trend_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: Overbought + volume confirmation + trend confirmation
            elif overbought and volume_confirm and trend_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Williams %R returns to -50 level (mean reversion)
            if position == 1:  # Long position
                if exit_level:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if exit_level:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals