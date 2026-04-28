#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend + volume confirmation + ATR stoploss
# Long when price > Donchian(20) high AND price > 1w EMA50 AND volume > 1.5x 20-bar avg
# Short when price < Donchian(20) low AND price < 1w EMA50 AND volume > 1.5x 20-bar avg
# Exit when price crosses Donchian(10) midpoint OR volume drops below average
# Target: 7-25 trades/year via tight breakout conditions with trend and volume filters
# Works in both bull and bear markets by requiring 1w EMA50 trend alignment

name = "1d_Donchian20_1wEMA50_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Calculate Donchian channels on 1d data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Donchian(20) for breakout signals
    donchian_high_20 = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # Donchian(10) for exit signals (midpoint)
    donchian_high_10 = high_series.rolling(window=10, min_periods=10).max().values
    donchian_low_10 = low_series.rolling(window=10, min_periods=10).min().values
    donchian_mid_10 = (donchian_high_10 + donchian_low_10) / 2.0
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high_20[i]) or 
            np.isnan(donchian_low_20[i]) or np.isnan(donchian_mid_10[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_50_1w_aligned[i]
        price = close[i]
        dc_high_20 = donchian_high_20[i]
        dc_low_20 = donchian_low_20[i]
        dc_mid_10 = donchian_mid_10[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price > Donchian(20) high AND price > 1w EMA50 AND volume confirmation
            if price > dc_high_20 and price > ema_trend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price < Donchian(20) low AND price < 1w EMA50 AND volume confirmation
            elif price < dc_low_20 and price < ema_trend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price crosses Donchian(10) midpoint OR no volume
            if price < dc_mid_10 or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price crosses Donchian(10) midpoint OR no volume
            if price > dc_mid_10 or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals