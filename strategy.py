#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Chop_Filter_v2
Hypothesis: 1d timeframe strategy using KAMA trend direction (bullish when price > KAMA, bearish when price < KAMA), combined with RSI mean-reversion (RSI < 30 long, RSI > 70 short) and chop filter (Choppiness Index > 61.8 for range conditions). Trades only when KAMA direction aligns with RSI extreme in choppy markets to capture mean reversions within larger trends. Designed for low trade frequency (<25/year) to minimize fee impact and work in both bull and bear regimes.
"""

name = "1d_KAMA_Direction_RSI_Chop_Filter_v2"
timeframe = "1d"
leverage = 1.0

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
    
    # Get 1w data for trend filter (strong trend avoidance in choppy markets)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w ADX for trend strength (avoid strong trends, favor ranging)
    # ADX calculation: +DI, -DI, DX
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smoothed values
        atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False).mean().values
        
        # DI values
        plus_di = 100 * dm_plus_smooth / atr
        minus_di = 100 * dm_minus_smooth / atr
        
        # DX and ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        
        return adx, plus_di, minus_di
    
    adx_1w, _, _ = calculate_adx(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def calculate_kama(close, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, k=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)  # This needs fixing
        
        # Proper ER calculation
        er = np.zeros_like(close)
        for i in range(period, len(close)):
            if i >= period:
                price_change = np.abs(close[i] - close[i-period])
                sum_abs_diff = np.sum(np.abs(np.diff(close[i-period:i+1])))
                if sum_abs_diff != 0:
                    er[i] = price_change / sum_abs_diff
                else:
                    er[i] = 0
        
        # Smoothing constants
        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA calculation
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        
        return kama
    
    # Calculate KAMA with proper ER
    kama = np.zeros_like(close)
    kama[:] = np.nan
    er = np.zeros_like(close)
    
    lookback = 10
    for i in range(lookback, len(close)):
        price_change = np.abs(close[i] - close[i-lookback])
        sum_abs_diff = 0
        for j in range(i-lookback+1, i+1):
            sum_abs_diff += np.abs(close[j] - close[j-1])
        if sum_abs_diff > 0:
            er[i] = price_change / sum_abs_diff
        else:
            er[i] = 0
    
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    for i in range(lookback, len(close)):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        if np.isnan(kama[i-1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    # Initialize first values
    for i in range(lookback):
        kama[i] = close[i]
    
    # Calculate RSI
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Calculate Choppiness Index
    def calculate_chop(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Absolute price change over period
        atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
        sum_tr = np.zeros_like(close)
        for i in range(period, len(close)):
            sum_tr[i] = np.sum(tr[i-period+1:i+1])
        
        # Max/min close over period
        max_close = np.zeros_like(close)
        min_close = np.zeros_like(close)
        for i in range(period-1, len(close)):
            max_close[i] = np.max(close[i-period+1:i+1])
            min_close[i] = np.min(close[i-period+1:i+1])
        
        # Chop calculation
        chop = np.zeros_like(close)
        for i in range(period, len(close)):
            if sum_tr[i] > 0 and max_close[i] != min_close[i]:
                chop[i] = 100 * np.log10(sum_tr[i] / (max_close[i] - min_close[i])) / np.log10(period)
            else:
                chop[i] = 50  # neutral
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    # Align indicators to lower timeframe (though we're on 1d, this ensures proper alignment)
    # Since we're on 1d timeframe, we'll use the values directly but ensure proper calculation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any key values are NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Conditions:
        # 1. Choppy market: Choppiness Index > 61.8 (ranging market)
        # 2. Weak trend: ADX < 25 (avoid strong trends)
        # 3. KAMA direction: price > KAMA = bullish bias, price < KAMA = bearish bias
        # 4. RSI extremes: RSI < 30 (oversold) for long, RSI > 70 (overbought) for short
        
        is_choppy = chop[i] > 61.8
        is_weak_trend = adx_1w_aligned[i] < 25
        
        if is_choppy and is_weak_trend:
            if position == 0:
                # LONG: Price above KAMA (bullish bias) + RSI oversold
                if close[i] > kama[i] and rsi[i] < 30:
                    signals[i] = 0.25
                    position = 1
                # SHORT: Price below KAMA (bearish bias) + RSI overbought
                elif close[i] < kama[i] and rsi[i] > 70:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # EXIT LONG: Price crosses below KAMA OR RSI overbought
                if close[i] < kama[i] or rsi[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # EXIT SHORT: Price crosses above KAMA OR RSI oversold
                if close[i] > kama[i] or rsi[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # Not in choppy/weak trend condition - stay flat
            signals[i] = 0.0
            position = 0
    
    return signals