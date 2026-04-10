#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and ADX regime filter
# - Primary: 1d price breaking above 20-period high or below 20-period low
# - HTF: 1w volume confirmation (current volume > 1.8x 20-period MA) + ADX > 20 for trend strength
# - Long: Breakout above upper band + volume confirmation + ADX > 20
# - Short: Breakout below lower band + volume confirmation + ADX > 20
# - Exit: Price returns to midpoint of Donchian channel
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Donchian captures breakouts, volume confirms momentum, ADX filters ranging markets
# - Target: 20-80 total trades over 4 years (5-20/year) to stay within fee drag limits

name = "1d_1w_donchian_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough data for Donchian and ADX
        return np.zeros(n)
    
    # Pre-compute 1d data
    close_1d = prices['close'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    volume_1d = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate Donchian channel (20-period) using previous 20 weeks' data
    period = 20
    # Upper band: highest high over previous 20 periods
    upper_channel = pd.Series(high_1w).rolling(window=period, min_periods=period).max().values
    # Lower band: lowest low over previous 20 periods
    lower_channel = pd.Series(low_1w).rolling(window=period, min_periods=period).min().values
    # Middle band: average of upper and lower
    middle_channel = (upper_channel + lower_channel) / 2.0
    
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
    period_adx = 14
    alpha = 1.0 / period_adx
    
    atr = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().values / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().values
    
    # Calculate 1w volume moving average (20-period) for volume confirmation
    volume_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 1d timeframe
    upper_channel_aligned = align_htf_to_ltf(prices, df_1w, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1w, lower_channel)
    middle_channel_aligned = align_htf_to_ltf(prices, df_1w, middle_channel)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    volume_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i]) or
            np.isnan(middle_channel_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(volume_ma_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1w volume (aligned to 1d)
        volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
        
        # Volume confirmation: current 1w volume > 1.8x 20-period MA
        volume_confirm = volume_1w_aligned[i] > 1.8 * volume_ma_20_1w_aligned[i]
        
        # ADX trend filter: ADX > 20 indicates sufficient trend strength
        trend_confirm = adx_aligned[i] > 20.0
        
        # Donchian breakout conditions
        breakout_long = close_1d[i] > upper_channel_aligned[i]
        breakout_short = close_1d[i] < lower_channel_aligned[i]
        
        # Exit conditions: Price returns to middle of Donchian channel
        exit_long = close_1d[i] < middle_channel_aligned[i]
        exit_short = close_1d[i] > middle_channel_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Breakout above upper band + volume confirmation + trend confirmation
            if breakout_long and volume_confirm and trend_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: Breakout below lower band + volume confirmation + trend confirmation
            elif breakout_short and volume_confirm and trend_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to middle channel
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