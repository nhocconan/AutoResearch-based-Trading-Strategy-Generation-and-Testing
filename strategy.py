#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1w volume confirmation and ADX regime filter
# - Primary: 12h price breaking above Donchian(20) high or below Donchian(20) low
# - HTF: 1w volume confirmation (current volume > 1.3x 50-period MA) + ADX > 20 for trend strength
# - Long: Breakout above Donchian high + volume confirmation + ADX > 20
# - Short: Breakout below Donchian low + volume confirmation + ADX > 20
# - Exit: Price returns to Donchian midpoint
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Donchian channels capture breakouts, volume confirms momentum, ADX filters weak trends
# - Target: 80-160 trades over 4 years (20-40/year) to stay within fee drag limits

name = "12h_1w_donchian_breakout_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:  # Need enough data for volume MA and ADX
        return np.zeros(n)
    
    # Pre-compute 12h data
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate Donchian channels (20-period) on 12h
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate ADX (1w) for trend strength
    # True Range
    tr1 = np.abs(np.roll(high_1w, 1) - np.roll(low_1w, 1))
    tr2 = np.abs(np.roll(high_1w, 1) - np.roll(close_1w, 1))
    tr3 = np.abs(np.roll(low_1w, 1) - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.roll(high_1w, 1) - high_1w
    down_move = low_1w - np.roll(low_1w, 1)
    
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
    
    # Calculate 1w volume moving average (50-period) for volume confirmation
    volume_ma_50_1w = pd.Series(volume_1w).rolling(window=50, min_periods=50).mean().values
    
    # Align all HTF indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close_12h}), donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close_12h}), donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close_12h}), donchian_mid)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    volume_ma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(volume_ma_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1w volume (aligned to 12h)
        volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
        
        # Volume confirmation: current 1w volume > 1.3x 50-period MA
        volume_confirm = volume_1w_aligned[i] > 1.3 * volume_ma_50_1w_aligned[i]
        
        # ADX trend filter: ADX > 20 indicates sufficient trend strength
        trend_confirm = adx_aligned[i] > 20.0
        
        # Donchian breakout conditions
        breakout_long = close_12h[i] > donchian_high_aligned[i]
        breakout_short = close_12h[i] < donchian_low_aligned[i]
        
        # Exit conditions: Price returns to Donchian midpoint
        exit_long = close_12h[i] < donchian_mid_aligned[i]
        exit_short = close_12h[i] > donchian_mid_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Breakout above Donchian high + volume confirmation + trend confirmation
            if breakout_long and volume_confirm and trend_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: Breakout below Donchian low + volume confirmation + trend confirmation
            elif breakout_short and volume_confirm and trend_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to Donchian midpoint
            if position == 1:  # Long position
                if exit_long:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals