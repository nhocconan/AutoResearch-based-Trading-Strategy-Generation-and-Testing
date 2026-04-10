#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and ADX regime filter
# - Primary: 4h price breaking above/below 20-period Donchian channels
# - HTF: 1d volume confirmation (current volume > 1.5x 20-period MA) + ADX > 25 for trend strength
# - Long: Breakout above upper Donchian + volume confirmation + ADX > 25
# - Short: Breakout below lower Donchian + volume confirmation + ADX > 25
# - Exit: Price returns to opposite Donchian level (long exits at lower, short exits at upper)
# - Position sizing: 0.30 (discrete level to minimize fee churn)
# - Works in bull/bear: Donchian captures breakouts, volume confirms conviction, ADX filters ranging markets
# - Target: 19-50 trades/year over 4 years (75-200 total) to stay within fee drag limits

name = "4h_1d_donchian_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for indicators
        return np.zeros(n)
    
    # Pre-compute 4h data
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    upper_donchian = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_donchian = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ADX (14-period) for trend strength
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
    period = 14
    alpha = 1.0 / period
    
    atr = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().values / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().values
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup period
        # Skip if any required data is invalid
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume (aligned to 4h)
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_confirm = volume_1d_aligned[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        # ADX trend filter: ADX > 25 indicates sufficient trend strength
        trend_confirm = adx_aligned[i] > 25.0
        
        # Donchian breakout conditions
        breakout_long = close_4h[i] > upper_donchian[i]
        breakout_short = close_4h[i] < lower_donchian[i]
        
        # Exit conditions: Price returns to opposite Donchian level
        exit_long = close_4h[i] < lower_donchian[i]
        exit_short = close_4h[i] > upper_donchian[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Breakout above upper Donchian + volume confirmation + trend confirmation
            if breakout_long and volume_confirm and trend_confirm:
                position = 1
                signals[i] = 0.30
            # Short entry: Breakout below lower Donchian + volume confirmation + trend confirmation
            elif breakout_short and volume_confirm and trend_confirm:
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to opposite Donchian level
            if position == 1:  # Long position
                if exit_long:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30
            else:  # position == -1 (Short position)
                if exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30
    
    return signals