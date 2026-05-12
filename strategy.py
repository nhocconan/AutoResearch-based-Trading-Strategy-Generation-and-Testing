# 12h_KAMA_Adaptive_Trend
# Hypothesis: On 12h timeframe, use KAMA to detect adaptive trend direction with dynamic smoothing based on market efficiency.
# Go long when KAMA slope is positive and price > KAMA, short when slope negative and price < KAMA.
# Filter with 1d ADX > 20 to ensure trending conditions and reduce whipsaws in ranging markets.
# Uses volume confirmation (volume > 1.5x 24-period average) to avoid low-liquidity breakouts.
# Targets 15-25 trades per year to minimize fee drag and improve generalization across bull/bear markets.
# Uses discrete position sizing (0.25) to minimize churn and respects all MTF data loading rules.

name = "12h_KAMA_Adaptive_Trend"
timeframe = "12h"
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

    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # KAMA (Kaufman Adaptive Moving Average) parameters
    kama_period = 10
    fast_ema = 2
    slow_ema = 30

    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, k= kama_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
    # Properly calculate volatility as sum of absolute changes over kama_period
    volatility = np.zeros_like(close)
    for i in range(kama_period, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-kama_period:i+1])))
    
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Calculate Smoothing Constant (SC)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Calculate KAMA slope (1-period change)
    kama_slope = np.diff(kama, prepend=kama[0])

    # 1d ADX for trend strength filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    def smooth_wilder(arr, period):
        result = np.zeros_like(arr)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr_smooth = smooth_wilder(tr, 14)
    plus_dm_smooth = smooth_wilder(plus_dm, 14)
    minus_dm_smooth = smooth_wilder(minus_dm, 14)
    
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 
                  0)
    adx = smooth_wilder(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)

    # Volume confirmation: volume > 1.5x 24-period average (approx 12 hours)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required value is NaN or invalid
        if (np.isnan(kama[i]) or np.isnan(kama_slope[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price > KAMA + positive KAMA slope + ADX > 20 + volume confirmation
            if (close[i] > kama[i] and 
                kama_slope[i] > 0 and 
                adx_aligned[i] > 20 and
                volume[i] > vol_avg_24[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < KAMA + negative KAMA slope + ADX > 20 + volume confirmation
            elif (close[i] < kama[i] and 
                  kama_slope[i] < 0 and 
                  adx_aligned[i] > 20 and
                  volume[i] > vol_avg_24[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price < KAMA OR KAMA slope turns negative OR ADX weakens
            if close[i] < kama[i] or kama_slope[i] < 0 or adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price > KAMA OR KAMA slope turns positive OR ADX weakens
            if close[i] > kama[i] or kama_slope[i] > 0 or adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals