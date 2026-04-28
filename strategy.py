#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high, price > 1w EMA50, volume > 1.5x 20-bar average
# Short when price breaks below Donchian(20) low, price < 1w EMA50, volume > 1.5x 20-bar average
# Exit when price retracs to Donchian(20) midpoint or opposite breakout occurs
# Uses 1d timeframe targeting 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
# Works in bull markets via breakouts with trend and in bear markets via breakdowns with trend.

name = "1d_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20)  # EMA50, Donchian20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        curr_high = donchian_high[i]
        curr_low = donchian_low[i]
        curr_mid = donchian_mid[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high, price above 1w EMA50, volume spike
            if price > curr_high and price > ema_50_1w_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below Donchian low, price below 1w EMA50, volume spike
            elif price < curr_low and price < ema_50_1w_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on retracement to midpoint or opposite breakout
            # Exit when price retracs to Donchian midpoint or breaks below Donchian low
            if price < curr_mid or price < curr_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on retracement to midpoint or opposite breakout
            # Exit when price retracs to Donchian midpoint or breaks above Donchian high
            if price > curr_mid or price > curr_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals