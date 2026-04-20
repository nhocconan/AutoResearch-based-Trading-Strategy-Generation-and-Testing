#!/usr/bin/env python3
"""
1d_KAMA_Direction_Plus_RSI_With_Chop_Filter
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with RSI(14) for momentum confirmation and Choppiness Index for regime filtering.
Enter long when KAMA trends up, RSI > 50 (bullish momentum), and market is not too choppy (CHOP < 61.8).
Enter short when KAMA trends down, RSI < 50 (bearish momentum), and market is not too choppy.
Exit when any condition fails. Uses 1-week ADX > 25 as additional trend strength filter.
Designed for low trade frequency (~10-25 trades/year) to minimize fee drag and work in both bull and bear markets.
"""

name = "1d_KAMA_Direction_Plus_RSI_With_Chop_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for indicators (same as primary but we need it for calculations)
    # For 1d timeframe, we can use prices directly for some calculations
    # But we still need 1w data for ADX filter
    
    # Get 1w data ONCE before loop for ADX filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX(14) for trend strength filter
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        # Plus Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth with Wilder's smoothing (EMA with alpha=1/period)
        def wilder_smoothing(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            # First value is simple average
            if len(data) >= period:
                result[period-1] = np.nansum(data[:period]) / period
                for i in range(period, len(data)):
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        atr = wilder_smoothing(tr, period)
        plus_di = 100 * wilder_smoothing(plus_dm, period) / atr
        minus_di = 100 * wilder_smoothing(minus_dm, period) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilder_smoothing(dx, period)
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate KAMA(10, 2, 30) on daily close
    def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
        # Efficiency Ratio
        change = np.abs(close - np.roll(close, er_period))
        volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if hasattr(np, 'sum') else np.abs(np.diff(close, n=1)).sum()
        # For array, we need to calculate volatility per point
        volatility_arr = np.zeros_like(close)
        for i in range(er_period, len(close)):
            volatility_arr[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
        volatility_arr[:er_period] = 1.0  # Avoid division by zero
        er = change / volatility_arr
        er[np.isnan(er)] = 0
        er[volatility_arr == 0] = 0
        
        # Smoothing constants
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        
        # KAMA
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Calculate RSI(14)
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # First average
        if len(close) > period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            
            # Smooth subsequent values
            for i in range(period+1, len(close)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate Choppiness Index(14)
    def calculate_choppiness(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # ATR (simple average for CHOP calculation)
        atr = np.zeros_like(close)
        for i in range(period, len(tr)):
            atr[i] = np.mean(tr[i-period+1:i+1])
        # For first period values, we'll calculate as we go
        
        # Sum of ATR over period
        sum_atr = np.zeros_like(close)
        for i in range(period-1, len(close)):
            sum_atr[i] = np.sum(atr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        for i in range(period-1, len(close)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
        
        # Choppiness Index
        chop = np.zeros_like(close)
        for i in range(period-1, len(close)):
            if highest_high[i] != lowest_low[i]:
                log_val = np.log10(sum_atr[i] / (highest_high[i] - lowest_low[i]))
                chop[i] = 100 * log_val / np.log10(period)
            else:
                chop[i] = 50  # Neutral when no range
        return chop
    
    # Calculate indicators
    kama = calculate_kama(close, 10, 2, 30)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    
    # KAMA direction: 1 if rising, -1 if falling
    kama_dir = np.zeros_like(close)
    kama_dir[1:] = np.where(kama[1:] > kama[:-1], 1, -1)
    
    # Session filter: avoid low-liquidity hours (optional for daily)
    # For daily, we can use all hours or filter to active trading hours
    # We'll use a simple time filter: avoid weekends if needed, but daily bars are fine
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or np.isnan(kama_dir[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA up, RSI > 50, CHOP < 61.8 (not too choppy), ADX > 25
            if (kama_dir[i] == 1 and rsi[i] > 50 and chop[i] < 61.8 and adx_1w_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI < 50, CHOP < 61.8, ADX > 25
            elif (kama_dir[i] == -1 and rsi[i] < 50 and chop[i] < 61.8 and adx_1w_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: any condition fails
            if (kama_dir[i] != 1 or rsi[i] <= 50 or chop[i] >= 61.8 or adx_1w_aligned[i] <= 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: any condition fails
            if (kama_dir[i] != -1 or rsi[i] >= 50 or chop[i] >= 61.8 or adx_1w_aligned[i] <= 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals