#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h KAMA trend with 4h RSI mean reversion and volume confirmation
# - Uses 12h KAMA (adaptive trend) to identify primary trend direction
# - Uses 4h RSI(14) for mean-reversion entries in trending markets
# - Uses 4h volume spike to confirm institutional participation
# - Enters long when: 12h KAMA rising AND 4h RSI < 30 AND volume spike
# - Enters short when: 12h KAMA falling AND 4h RSI > 70 AND volume spike
# - Exits when RSI returns to neutral zone (40-60) or opposite extreme
# - Designed to capture pullbacks in strong trends with institutional validation
# - Target: 100-200 total trades over 4 years (25-50/year) with 0.25 position sizing

name = "4h_12hKAMA_4hRSI_Volume_MeanReversion"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for KAMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h KAMA ( Kaufman Adaptive Moving Average )
    close_12h = df_12h['close'].values
    
    # Efficiency Ratio (ER) calculation
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility = np.sum(np.abs(np.diff(close_12h)), axis=0)
    # For simplicity, we'll use a rolling approach
    er = np.zeros_like(close_12h)
    for i in range(1, len(close_12h)):
        if i >= 10:  # 10-period lookback for ER
            direction = np.abs(close_12h[i] - close_12h[i-10])
            volatility_sum = np.sum(np.abs(np.diff(close_12h[i-10:i+1])))
            if volatility_sum > 0:
                er[i] = direction / volatility_sum
            else:
                er[i] = 0
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # KAMA direction (rising/falling)
    kama_rising = kama > np.roll(kama, 1)
    kama_falling = kama < np.roll(kama, 1)
    kama_rising[0] = False
    kama_falling[0] = False
    
    # Align 12h KAMA to 4h timeframe
    kama_rising_4h = align_htf_to_ltf(prices, df_12h, kama_rising)
    kama_falling_4h = align_htf_to_ltf(prices, df_12h, kama_falling)
    
    # 4h RSI(14) for mean reversion
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter (4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama_rising_4h[i]) or np.isnan(kama_falling_4h[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising, RSI oversold, volume spike
            if kama_rising_4h[i] and rsi[i] < 30 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, RSI overbought, volume spike
            elif kama_falling_4h[i] and rsi[i] > 70 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI returns to neutral or overbought
            if rsi[i] >= 40:  # Exit when RSI reaches neutral or better
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI returns to neutral or oversold
            if rsi[i] <= 60:  # Exit when RSI reaches neutral or worse
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals