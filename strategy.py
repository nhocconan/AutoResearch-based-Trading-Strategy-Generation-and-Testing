#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA21 trend filter + volume confirmation
# - Long when price breaks above Donchian(20) high + price > 1w EMA21 (uptrend) + volume > 1.5x average
# - Short when price breaks below Donchian(20) low + price < 1w EMA21 (downtrend) + volume > 1.5x average
# - Exit when price crosses back through Donchian midpoint or trend changes
# - Designed for 1d timeframe with selective entries to avoid overtrading
# - Target: 7-25 trades per year per symbol (30-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for EMA21 calculation
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(21) on 1w timeframe
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 1w EMA21 to 1d timeframe
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate Donchian(20) on 1d timeframe
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after Donchian warmup
        # Skip if NaN in indicators
        if np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or np.isnan(ema_21_1w_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        donchian_high = highest_high_20[i]
        donchian_low = lowest_low_20[i]
        donchian_mid_val = donchian_mid[i]
        ema_trend = ema_21_1w_aligned[i]
        vol_avg = avg_volume[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high + uptrend + volume confirmation
            if price > donchian_high and price > ema_trend and vol > 1.5 * vol_avg:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + downtrend + volume confirmation
            elif price < donchian_low and price < ema_trend and vol > 1.5 * vol_avg:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian midpoint or trend turns bearish
            if price < donchian_mid_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian midpoint or trend turns bullish
            if price > donchian_mid_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA21_VolumeFilter"
timeframe = "1d"
leverage = 1.0