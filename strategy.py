#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour ADX + RSI reversal with 4h EMA trend filter and volume confirmation
# Long when ADX > 25 (trending) AND RSI < 30 (oversold) AND price > 4h EMA200 AND volume > 1.5x avg
# Short when ADX > 25 (trending) AND RSI > 70 (overbought) AND price < 4h EMA200 AND volume > 1.5x avg
# Exit when RSI crosses back to neutral (40-60 range)
# Uses ADX to filter for trending markets, RSI for mean reversion entries, EMA for trend alignment
# Target: 60-150 total trades over 4 years (15-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for EMA200 trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate ADX (14-period)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = 0.9 * atr[i-1] + 0.1 * tr[i]  # Wilder's smoothing
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    for i in range(1, n):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm[i] / atr[i]
            minus_di[i] = 100 * minus_dm[i] / atr[i]
            if plus_di[i] + minus_di[i] > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = np.zeros(n)
    adx_start = 14
    if n > adx_start * 2:
        adx[adx_start] = np.mean(dx[1:adx_start+1])
        for i in range(adx_start+1, n):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Calculate RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros(n)
    rsi = np.zeros(n)
    for i in range(n):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100 if avg_gain[i] > 0 else 50
    
    # Calculate 4h EMA200 for trend filter
    close_4h = df_4h['close'].values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = max(50, 200)  # EMA200 needs 200 periods
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema200_4h_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: trending + oversold + above 4h EMA200 + volume confirmation
            if (adx[i] > 25 and rsi[i] < 30 and price > ema200_4h_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: trending + overbought + below 4h EMA200 + volume confirmation
            elif (adx[i] > 25 and rsi[i] > 70 and price < ema200_4h_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral territory (40-60)
            if rsi[i] >= 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral territory (40-60)
            if rsi[i] <= 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_ADX_RSI_Reversal_4hEMA200_Volume"
timeframe = "1h"
leverage = 1.0