#!/usr/bin/env python3
"""
1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation
Hypothesis: Daily Donchian(20) breakouts capture major trends. Combined with weekly EMA50 trend filter (bull/bear regime) and volume spike (>2.0x 20-bar vol MA) to confirm momentum. Works in bull markets via upside breakouts and bear markets via downside breakouts. Targeting 15-25 trades per year to minimize fee drag while maintaining edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 51:  # Need 50 for EMA + 1 for shift
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'])
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Donchian(20) levels (using prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:  # Need 20 for Donchian + 1 for shift
        return np.zeros(n)
    
    # Calculate Donchian levels from previous 20 days
    # Upper = max(high) over past 20 days, Lower = min(low) over past 20 days
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    donchian_upper = high_1d.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_1d.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 1d timeframe (already aligned, but keep for consistency)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Calculate 20-period volume MA for volume spike confirmation (1d)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA50, Donchian, and volume MA
    start_idx = max(51, 21, 20)  # 51 for EMA50 (50 + 1 for shift), 21 for Donchian (20 + 1 for shift), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50_val = ema_50_1w_aligned[i]
        upper_val = donchian_upper_aligned[i]
        lower_val = donchian_lower_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Trend filter: price above/below 1w EMA50
        price_above_ema = curr_close > ema_50_val
        price_below_ema = curr_close < ema_50_val
        
        if position == 0:
            # Long: break above upper Donchian + price above 1w EMA50 + volume confirmation
            long_signal = (curr_high > upper_val) and price_above_ema and volume_confirm
            # Short: break below lower Donchian + price below 1w EMA50 + volume confirmation
            short_signal = (curr_low < lower_val) and price_below_ema and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below upper Donchian OR price crosses below 1w EMA50
            if (curr_close < upper_val) or (curr_close < ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above lower Donchian OR price crosses above 1w EMA50
            if (curr_close > lower_val) or (curr_close > ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0