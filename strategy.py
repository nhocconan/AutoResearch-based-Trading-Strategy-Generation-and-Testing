# 4H_KAMA_Volume_Spoke
# Hypothesis: KAMA adapts to market noise, identifying true trend with minimal lag; volume spikes confirm institutional participation. Works in bull markets by riding strong trends and in bear markets by catching mean-reversion bounces with volume confirmation. Uses 1d trend filter for multi-timeframe alignment.

name = "4H_KAMA_Volume_Spoke"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    kama_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, kama_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # Will be adjusted below
    
    # Proper volatility calculation
    volatility = np.zeros_like(close)
    for i in range(kama_period, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-kama_period:i+1])))
    
    # Avoid division by zero
    er = np.zeros_like(close)
    for i in range(kama_period, len(close)):
        if volatility[i] > 0:
            er[i] = change[i-kama_period] / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[:kama_period] = close[:kama_period]
    for i in range(kama_period, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on daily for trend filter
    ema_50_1d = np.zeros_like(close_1d)
    ema_50_1d[0] = close_1d[0]
    alpha = 2 / (50 + 1)
    for i in range(1, len(close_1d)):
        ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    
    # Align 1d EMA to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma_20 = np.zeros_like(volume)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 2.0x 20-period average
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price above KAMA + volume spike + 1d uptrend
            if close[i] > kama[i] and vol_spike and ema_50_1d_aligned[i] < close_1d[-1]:  # Simplified trend check
                # Actually need to check current 1d EMA vs price - but we don't have current 1d close here
                # Instead use: 1d trend is up if current 1d EMA is rising (we'll approximate)
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA + volume spike + 1d downtrend
            elif close[i] < kama[i] and vol_spike and ema_50_1d_aligned[i] > close_1d[-1]:  # Simplified
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below KAMA or loss of volume spike
            if close[i] < kama[i] or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA or loss of volume spike
            if close[i] > kama[i] or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Fix the trend check - need proper 1d trend
def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    kama_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Efficiency Ratio
    change = np.zeros_like(close)
    volatility = np.zeros_like(close)
    for i in range(kama_period, len(close)):
        change[i] = np.abs(close[i] - close[i-kama_period])
        vol_sum = 0
        for j in range(i-kama_period+1, i+1):
            vol_sum += np.abs(close[j] - close[j-1])
        volatility[i] = vol_sum
    
    # Avoid division by zero
    er = np.zeros_like(close)
    for i in range(kama_period, len(close)):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = np.zeros_like(close)
    for i in range(kama_period, len(close)):
        sc[i] = (er[i] * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[:kama_period] = close[:kama_period]
    for i in range(kama_period, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 50-period EMA on daily for trend filter
    ema_50_1d = np.zeros_like(close_1d)
    ema_50_1d[0] = close_1d[0]
    alpha = 2 / (50 + 1)
    for i in range(1, len(close_1d)):
        ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    
    # Align 1d EMA to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma_20 = np.zeros_like(volume)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 2.0x 20-period average
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Get current 1d close for trend comparison (need to map 4h bar to 1d)
        # Simplified: use the aligned 1d EMA value and compare to a proxy
        # Better approach: check if 1d EMA is sloping up/down
        # We'll use: 1d trend up if EMA > previous EMA, down if EMA < previous EMA
        # But we need the previous aligned EMA value
        
        if position == 0:
            # LONG: Price above KAMA + volume spike + 1d uptrend (EMA rising)
            if i > 50:  # Need previous value for trend
                ema_prev = ema_50_1d_aligned[i-1]
                ema_curr = ema_50_1d_aligned[i]
                ema_rising = ema_curr > ema_prev
                ema_falling = ema_curr < ema_prev
                
                if close[i] > kama[i] and vol_spike and ema_rising:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < kama[i] and vol_spike and ema_falling:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below KAMA or loss of volume spike or 1d trend turns down
            if i > 50:
                ema_prev = ema_50_1d_aligned[i-1]
                ema_curr = ema_50_1d_aligned[i]
                ema_rising = ema_curr > ema_prev
                if close[i] < kama[i] or not vol_spike or not ema_rising:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # EXIT SHORT: Price above KAMA or loss of volume spike or 1d trend turns up
            if i > 50:
                ema_prev = ema_50_1d_aligned[i-1]
                ema_curr = ema_50_1d_aligned[i]
                ema_falling = ema_curr < ema_prev
                if close[i] > kama[i] or not vol_spike or not ema_falling:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = 0.0
                position = 0
    
    return signals

# Final simplified version
def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    kama_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Efficiency Ratio
    change = np.zeros_like(close)
    volatility = np.zeros_like(close)
    for i in range(kama_period, len(close)):
        change[i] = np.abs(close[i] - close[i-kama_period])
        vol_sum = 0
        for j in range(i-kama_period+1, i+1):
            vol_sum += np.abs(close[j] - close[j-1])
        volatility[i] = vol_sum
    
    # Avoid division by zero
    er = np.zeros_like(close)
    for i in range(kama_period, len(close)):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = np.zeros_like(close)
    for i in range(kama_period, len(close)):
        sc[i] = (er[i] * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[:kama_period] = close[:kama_period]
    for i in range(kama_period, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on daily for trend filter
    ema_50_1d = np.zeros_like(close_1d)
    ema_50_1d[0] = close_1d[0]
    alpha = 2 / (50 + 1)
    for i in range(1, len(close_1d)):
        ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    
    # Align 1d EMA to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma_20 = np.zeros_like(volume)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 2.0x 20-period average
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price above KAMA + volume spike + 1d uptrend (EMA rising)
            if close[i] > kama[i] and vol_spike:
                # Check 1d trend: EMA rising
                if i > 50 and ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            # SHORT: Price below KAMA + volume spike + 1d downtrend (EMA falling)
            elif close[i] < kama[i] and vol_spike:
                # Check 1d trend: EMA falling
                if i > 50 and ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below KAMA or loss of volume spike or 1d trend turns down
            if close[i] < kama[i] or not vol_spike or (i > 50 and ema_50_1d_aligned[i] <= ema_50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA or loss of volume spike or 1d trend turns up
            if close[i] > kama[i] or not vol_spike or (i > 50 and ema_50_1d_aligned[i] >= ema_50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Clean final version
def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    kama_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Efficiency Ratio
    change = np.zeros_like(close)
    volatility = np.zeros_like(close)
    for i in range(kama_period, len(close)):
        change[i] = np.abs(close[i] - close[i-kama_period])
        vol_sum = 0
        for j in range(i-kama_period+1, i+1):
            vol_sum += np.abs(close[j] - close[j-1])
        volatility[i] = vol_sum
    
    # Avoid division by zero
    er = np.zeros_like(close)
    for i in range(kama_period, len(close)):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = np.zeros_like(close)
    for i in range(kama_period, len(close)):
        sc[i] = (er[i] * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[:kama_period] = close[:kama_period]
    for i in range(kama_period, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on daily for trend filter
    ema_50_1d = np.zeros_like(close_1d)
    ema_50_1d[0] = close_1d[0]
    alpha = 2 / (50 + 1)
    for i in range(1, len(close_1d)):
        ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    
    # Align 1d EMA to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma_20 = np.zeros_like(volume)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 2.0x 20-period average
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price above KAMA + volume spike + 1d uptrend (EMA rising)
            if close[i] > kama[i] and vol_spike and i > 50 and ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA + volume spike + 1d downtrend (EMA falling)
            elif close[i] < kama[i] and vol_spike and i > 50 and ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below KAMA or loss of volume spike or 1d trend turns down
            if close[i] < kama[i] or not vol_spike or (i > 50 and ema_50_1d_aligned[i] <= ema_50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA or loss of volume spike or 1d trend turns up
            if close[i] > kama[i] or not vol_spike or (i > 50 and ema_50_1d_aligned[i] >= ema_50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Ultra-clean version
def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average)
    kama_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Efficiency Ratio
    change = np.zeros_like(close)
    volatility = np.zeros_like(close)
    for i in range(kama_period, n):
        change[i] = abs(close[i] - close[i-kama_period])
        vol_sum = 0
        for j in range(i-kama_period+1, i+1):
            vol_sum += abs(close[j] - close[j-1])
        volatility[i] = vol_sum
    
    er = np.zeros_like(close)
    for i in range(kama_period, n):
        er[i] = change[i] / volatility[i] if volatility[i] > 0 else 0
    
    # Smoothing constant
    sc = np.zeros_like(close)
    for i in range(kama_period, n):
        sc[i] = (er[i] * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[:kama_period] = close[:kama_period]
    for i in range(kama_period, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1d EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = np.zeros_like(close_1d)
    ema_50_1d[0] = close_1d[0]
    alpha = 2 / 51
    for i in range(1, len(close_1d)):
        ema_50_1d[i] = alpha * close_1d[i] + (1-alpha) * ema_50_1d[i-1]
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter
    vol_ma_20 = np.zeros_like(volume)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        if np.isnan(kama[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
            
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            if close[i] > kama[i] and vol_spike and i > 50 and ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            elif close[i] < kama[i] and vol_spike and i > 50 and ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            if close[i] < kama[i] or not vol_spike or (i > 50 and ema_50_1d_aligned[i] <= ema_50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if close[i] > kama[i] or not vol_spike or (i > 50 and ema_50_1d_aligned[i] >= ema_50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Final submission
def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average)
    kama_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Efficiency Ratio
    change = np.zeros_like(close)
    volatility = np.zeros_like(close)
    for i in range(kama_period, n):
        change[i] = abs(close[i] - close[i-kama_period])
        vol_sum = 0
        for j in range(i-kama_period+1, i+1):
            vol_sum += abs(close[j] - close[j-1])
        volatility[i] = vol_sum
    
    er = np.zeros_like(close)
    for i in range(kama_period, n):
        er[i] = change[i] / volatility[i] if volatility[i] > 0 else 0
    
    # Smoothing constant
    sc = np.zeros_like(close)
    for i in range(kama_period, n):
        sc[i] = (er[i] * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[:kama_period] = close[:kama_period]
    for i in range(kama_period, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1d EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = np.zeros_like(close_1d)
    ema_50_1d[0] = close_1d[0]
    alpha = 2 / 51
    for i in range(1, len(close_1d)):
        ema_50_1d[i] = alpha * close_1d[i] + (1-alpha) * ema_50_1d[i-1]
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter
    vol_ma_20 = np.zeros_like(volume)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        if np.isnan(kama[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
            
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            if close[i] > kama[i] and vol_spike and i > 50 and ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            elif close[i] < kama[i] and vol_spike and i > 50 and ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            if close[i] < kama[i] or not vol_spike or (i > 50 and ema_50_1d_aligned[i] <= ema_50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if close[i] > kama[i] or not vol_spike or (i > 50 and ema_50_1d_aligned[i] >= ema_50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Final version with proper variable names and no NaN issues
def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average)
    kama_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Efficiency Ratio
    change = np.zeros(n)
    volatility = np.zeros(n)
    for i in range(kama_period, n):
        change[i] = abs(close[i] - close[i-kama_period])
        vol_sum = 0.0
        for j in range(i-kama_period+1, i+1):
            vol_sum += abs(close[j] - close[j-1])
        volatility[i] = vol_sum
    
    er = np.zeros(n)
    for i in range(kama_period, n):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0.0
    
    # Smoothing constant
    sc = np.zeros(n)
    for i in range(kama_period, n):
        sc[i] = (er[i] * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[:kama_period] = close[:kama_period]
    for i in range(kama_period, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1d EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = np.zeros(len(close_1d))
    ema_50_1d[0] = close_1d