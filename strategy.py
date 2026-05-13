# 4H_KAMA_RSI_CHOP_FILTER
# Hypothesis: Use KAMA trend direction on 4h combined with RSI(14) extremes and Choppiness Index(14) regime filter for mean reversion in ranging markets and trend continuation in trending markets. KAMA adapts to volatility, RSI identifies overbought/oversold, and Choppiness Index determines market regime. This combination should work in both bull and bear markets by adapting to regime.
# Timeframe: 4h, using 12h for HTF trend filter (EMA34)
# Entry: Long when KAMA rising, RSI<30, CHOP>61.8 (range). Short when KAMA falling, RSI>70, CHOP>61.8 (range). In trending markets (CHOP<38.2), follow KAMA direction with RSI>50 for long, RSI<50 for short.
# Exit: Opposite signal or trend change.
# Position sizing: 0.25 to limit drawdown and reduce trade frequency.

name = "4H_KAMA_RSI_CHOP_FILTER"
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

    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)

    # KAMA on 4h
    def calculate_kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)  # This needs fixing - will compute properly below
        # Better approach: calculate ER properly
        er = np.zeros_like(close)
        for i in range(length, len(close)):
            if i >= length:
                change_val = np.abs(close[i] - close[i-length])
                volatility_val = np.sum(np.abs(np.diff(close[i-length+1:i+1])))
                if volatility_val != 0:
                    er[i] = change_val / volatility_val
                else:
                    er[i] = 0
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama

    # Simpler: use EMA as proxy for adaptive trend, but we'll implement proper KAMA
    # Actually, let's use a simpler adaptive approach or use RSI-based adaptation
    # For now, use EMA20 for trend but with volatility adjustment concept
    # We'll implement a simplified version that captures the essence
    
    # Calculate ER properly
    change = np.zeros_like(close)
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        change[i] = np.abs(close[i] - close[i-1])
    
    # ER over 10 periods
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        price_change = np.abs(close[i] - close[i-10])
        sum_abs_change = np.sum(np.abs(np.diff(close[i-10:i+1])))
        if sum_abs_change > 0:
            er[i] = price_change / sum_abs_change
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2/(2+1)
    slow_sc = 2/(30+1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # Handle division by zero
    rsi = np.where(avg_loss == 0, 100, rsi)
    rsi = np.where(avg_gain == 0, 0, rsi)

    # Choppiness Index(14)
    def choppiness_index(high, low, close, length=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Sum of TR over period
        atr_sum = np.zeros_like(close)
        for i in range(length, len(close)):
            atr_sum[i] = np.sum(tr[i-length+1:i+1])
        
        # Highest high and lowest low over period
        max_high = np.zeros_like(close)
        min_low = np.zeros_like(close)
        for i in range(length-1, len(close)):
            max_high[i] = np.max(high[i-length+1:i+1])
            min_low[i] = np.min(low[i-length+1:i+1])
        
        # Chop calculation
        chop = np.zeros_like(close)
        for i in range(length, len(close)):
            if atr_sum[i] > 0 and (max_high[i] - min_low[i]) > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(length)
            else:
                chop[i] = 50  # neutral
        return chop

    chop = choppiness_index(high, low, close, 14)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # start after warmup period
        # Skip if any required value is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_34_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Range market conditions (CHOP > 61.8) - mean reversion
            if chop[i] > 61.8:
                # LONG: oversold (RSI < 30) and price above KAMA (bullish bias)
                if rsi[i] < 30 and close[i] > kama[i]:
                    signals[i] = 0.25
                    position = 1
                # SHORT: overbought (RSI > 70) and price below KAMA (bearish bias)
                elif rsi[i] > 70 and close[i] < kama[i]:
                    signals[i] = -0.25
                    position = -1
            # Trending market conditions (CHOP < 38.2) - trend following
            elif chop[i] < 38.2:
                # LONG: rising KAMA (uptrend) and RSI > 50 (bullish momentum)
                if i > 0 and kama[i] > kama[i-1] and rsi[i] > 50:
                    signals[i] = 0.25
                    position = 1
                # SHORT: falling KAMA (downtrend) and RSI < 50 (bearish momentum)
                elif i > 0 and kama[i] < kama[i-1] and rsi[i] < 50:
                    signals[i] = -0.25
                    position = -1
            # Transition zone (CHOP between 38.2 and 61.8) - use HTF trend filter
            else:
                # LONG: price above 12h EMA34 and RSI > 50
                if close[i] > ema_34_12h_aligned[i] and rsi[i] > 50:
                    signals[i] = 0.25
                    position = 1
                # SHORT: price below 12h EMA34 and RSI < 50
                elif close[i] < ema_34_12h_aligned[i] and rsi[i] < 50:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # EXIT LONG: opposite conditions
            # Exit if: RSI > 70 (overbought) or chop < 38.2 and kama falling (trend change)
            if rsi[i] > 70 or (chop[i] < 38.2 and i > 0 and kama[i] < kama[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: opposite conditions
            # Exit if: RSI < 30 (oversold) or chop < 38.2 and kama rising (trend change)
            if rsi[i] < 30 or (chop[i] < 38.2 and i > 0 and kama[i] > kama[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals