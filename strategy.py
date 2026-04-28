#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation
# Long when price breaks above Donchian(20) high AND price > 1w EMA50 AND volume > 1.5x 20-bar avg
# Short when price breaks below Donchian(20) low AND price < 1w EMA50 AND volume > 1.5x 20-bar avg
# Exit when price crosses the Donchian midpoint or trend filter fails
# Target: 7-25 trades/year via tight entry conditions and strong filters
# Works in both bull and bear markets by only trading in direction of 1w EMA50 trend

name = "1d_Donchian20_1wEMA50_Trend_VolumeFilter_v1"
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w close
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) channels on 1d data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Donchian high: 20-period rolling maximum
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    # Donchian low: 20-period rolling minimum
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    # Donchian midpoint: average of high and low channels
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need sufficient history for Donchian(20) and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_50 = ema_50_1w_aligned[i]
        price = close[i]
        upper = donch_high[i]
        lower = donch_low[i]
        mid = donch_mid[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Donchian high AND above 1w EMA50 AND volume confirmation
            if price > upper and price > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low AND below 1w EMA50 AND volume confirmation
            elif price < lower and price < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price crosses below Donchian mid or trend fails
            if price < mid or price < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price crosses above Donchian mid or trend fails
            if price > mid or price > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals