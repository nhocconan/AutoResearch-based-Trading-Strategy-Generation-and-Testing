#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and ADX regime filter
# - Primary: 4h price breaking above/below 20-period Donchian channels
# - HTF: 12h volume confirmation (current volume > 1.8x 20-period MA) + ADX > 20 for trend strength
# - Long: Breakout above upper channel + volume confirmation + ADX > 20
# - Short: Breakout below lower channel + volume confirmation + ADX > 20
# - Exit: Price returns to opposite channel (long exits at lower, short exits at upper)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Donchian captures breakouts, volume confirms momentum, ADX filters ranging markets
# - Target: 80-160 trades over 4 years (20-40/year) to stay within fee drag limits

name = "4h_12h_donchian_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough data for volume MA and ADX
        return np.zeros(n)
    
    # Pre-compute 4h data
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Pre-compute 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    # Upper channel: highest high over past 20 periods
    # Lower channel: lowest low over past 20 periods
    upper_channel = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h ADX for trend strength
    # True Range
    tr1 = np.abs(np.roll(high_12h, 1) - np.roll(low_12h, 1))
    tr2 = np.abs(np.roll(high_12h, 1) - np.roll(close_12h, 1))
    tr3 = np.abs(np.roll(low_12h, 1) - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.roll(high_12h, 1) - high_12h
    down_move = low_12h - np.roll(low_12h, 1)
    
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
    
    # Calculate 12h volume moving average (20-period) for volume confirmation
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 4h timeframe
    upper_channel_aligned = align_htf_to_ltf(prices, df_12h, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_12h, lower_channel)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    volume_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 12h volume (aligned to 4h)
        volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        
        # Volume confirmation: current 12h volume > 1.8x 20-period MA
        volume_confirm = volume_12h_aligned[i] > 1.8 * volume_ma_20_12h_aligned[i]
        
        # ADX trend filter: ADX > 20 indicates sufficient trend strength
        trend_confirm = adx_aligned[i] > 20.0
        
        # Donchian breakout conditions
        breakout_long = close_4h[i] > upper_channel_aligned[i]
        breakout_short = close_4h[i] < lower_channel_aligned[i]
        
        # Exit conditions: Price returns to opposite channel
        exit_long = close_4h[i] < lower_channel_aligned[i]  # Long exits at lower channel
        exit_short = close_4h[i] > upper_channel_aligned[i]  # Short exits at upper channel
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Breakout above upper channel + volume confirmation + trend confirmation
            if breakout_long and volume_confirm and trend_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: Breakout below lower channel + volume confirmation + trend confirmation
            elif breakout_short and volume_confirm and trend_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to opposite channel
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