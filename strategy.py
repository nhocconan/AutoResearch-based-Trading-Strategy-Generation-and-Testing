#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction + RSI + chop filter
# Uses KAMA (Kaufman Adaptive Moving Average) to capture trend direction,
# combined with RSI for momentum confirmation and Choppiness Index to filter range-bound markets.
# In trending markets (CHOP < 38.2), KAMA > RSI(50) signals long, KAMA < RSI(50) signals short.
# In ranging markets (CHOP > 61.8), strategy remains flat to avoid whipsaw.
# Weekly trend filter (1w EMA34) ensures alignment with higher timeframe momentum.
# Designed for low trade frequency (10-25 trades/year) with high win rate in both bull and bear markets.
name = "1d_KAMA_RSI_Chop_WeeklyEMA"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # KAMA (Kaufman Adaptive Moving Average) - 14-period
    def kama(data, period=14):
        # Efficiency Ratio
        change = np.abs(np.diff(data, n=period))
        volatility = np.sum(np.abs(np.diff(data)), axis=0)
        # Handle first period values
        er = np.full_like(data, np.nan, dtype=float)
        er[period:] = change[period-1:] / volatility[period-1:]
        # Smoothing constants
        sc = np.full_like(data, np.nan, dtype=float)
        sc[period:] = (er[period:] * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
        # Initialize KAMA
        kama_vals = np.full_like(data, np.nan, dtype=float)
        kama_vals[period-1] = np.mean(data[:period])
        # Calculate KAMA
        for i in range(period, len(data)):
            if not np.isnan(sc[i]):
                kama_vals[i] = kama_vals[i-1] + sc[i] * (data[i] - kama_vals[i-1])
            else:
                kama_vals[i] = kama_vals[i-1]
        return kama_vals
    
    kama_vals = kama(close, 14)
    
    # RSI (14-period)
    def rsi(data, period=14):
        delta = np.diff(data)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full_like(data, np.nan, dtype=float)
        avg_loss = np.full_like(data, np.nan, dtype=float)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(data)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.divide(avg_gain, avg_loss, out=np.full_like(data, np.nan, dtype=float), where=avg_loss!=0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_vals = rsi(close, 14)
    
    # Choppiness Index (14-period)
    def choppy(data_high, data_low, data_close, period=14):
        # True Range
        tr1 = data_high[1:] - data_low[1:]
        tr2 = np.abs(data_high[1:] - data_close[:-1])
        tr3 = np.abs(data_low[1:] - data_close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # Align with original index
        
        # Sum of TR over period
        tr_sum = np.full_like(data_close, np.nan, dtype=float)
        for i in range(period, len(data_close)):
            tr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        max_high = np.full_like(data_close, np.nan, dtype=float)
        min_low = np.full_like(data_close, np.nan, dtype=float)
        for i in range(period-1, len(data_close)):
            max_high[i] = np.max(data_high[i-period+1:i+1])
            min_low[i] = np.min(data_low[i-period+1:i+1])
        
        # Choppiness Index
        chop = np.full_like(data_close, np.nan, dtype=float)
        for i in range(period, len(data_close)):
            if tr_sum[i] > 0 and max_high[i] > min_low[i]:
                chop[i] = 100 * np.log10(tr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
        return chop
    
    chop_vals = choppy(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or np.isnan(chop_vals[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA > RSI(50) + CHOP < 38.2 (trending) + above weekly EMA
            if (kama_vals[i] > 50 and rsi_vals[i] > 50 and 
                chop_vals[i] < 38.2 and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA < RSI(50) + CHOP < 38.2 (trending) + below weekly EMA
            elif (kama_vals[i] < 50 and rsi_vals[i] < 50 and 
                  chop_vals[i] < 38.2 and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if KAMA < RSI(50) or CHOP > 61.8 (ranging) or below weekly EMA
            if (kama_vals[i] < 50) or (chop_vals[i] > 61.8) or (close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if KAMA > RSI(50) or CHOP > 61.8 (ranging) or above weekly EMA
            if (kama_vals[i] > 50) or (chop_vals[i] > 61.8) or (close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals