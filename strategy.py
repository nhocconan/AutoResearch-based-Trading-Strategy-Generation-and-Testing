#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d KAMA + RSI + Chop Filter
# Hypothesis: KAMA adapts to market noise, providing reliable trend direction.
# RSI(14) filters entries to avoid overbought/oversold extremes.
# Chop filter (Choppiness Index) avoids ranging markets where trends fail.
# Designed for 1d timeframe with low trade frequency (<25/year).
# Works in bull via KAMA uptrend + RSI < 70, in bear via KAMA downtrend + RSI > 30.
# Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for chop filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # KAMA (Kaufman Adaptive Moving Average) on daily close
    # ER (Efficiency Ratio) = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    # SC = [ER * (fastest - slowest) + slowest]^2
    # KAMA = prevKAMA + SC * (price - prevKAMA)
    def kama(data, er_period=10, fast=2, slow=30):
        if len(data) < er_period:
            return np.full_like(data, np.nan, dtype=float)
        # Calculate change and volatility
        change = np.abs(np.subtract(data[er_period:], data[:-er_period]))
        volatility = np.sum(np.abs(np.diff(data)), axis=0) if len(data) > 1 else 0
        # For array calculation, we need to compute volatility over rolling window
        er = np.full_like(data, np.nan, dtype=float)
        for i in range(er_period, len(data)):
            if np.sum(np.abs(np.diff(data[i-er_period:i+1]))) > 0:
                er[i] = np.abs(data[i] - data[i-er_period]) / np.sum(np.abs(np.diff(data[i-er_period:i+1])))
            else:
                er[i] = 0
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # Initialize KAMA
        kama_vals = np.full_like(data, np.nan, dtype=float)
        kama_vals[er_period] = np.mean(data[:er_period+1])  # Start with average
        for i in range(er_period+1, len(data)):
            if not np.isnan(sc[i]):
                kama_vals[i] = kama_vals[i-1] + sc[i] * (data[i] - kama_vals[i-1])
            else:
                kama_vals[i] = kama_vals[i-1]
        return kama_vals
    
    kama_vals = kama(close)
    
    # RSI(14) on daily close
    def rsi(data, period=14):
        if len(data) < period + 1:
            return np.full_like(data, np.nan, dtype=float)
        delta = np.diff(data)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full_like(data, np.nan, dtype=float)
        avg_loss = np.full_like(data, np.nan, dtype=float)
        # First average
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        # Subsequent averages (Wilder's smoothing)
        for i in range(period+1, len(data)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi_vals = np.full_like(data, np.nan, dtype=float)
        rsi_vals[period:] = 100 - (100 / (1 + rs[period:]))
        return rsi_vals
    
    rsi_vals = rsi(close)
    
    # Choppiness Index on weekly data
    def chop(data_high, data_low, data_close, period=14):
        if len(data_close) < period:
            return np.full_like(data_close, np.nan, dtype=float)
        # True Range
        tr1 = np.subtract(data_high[1:], data_low[1:])
        tr2 = np.abs(np.subtract(data_high[1:], data_close[:-1]))
        tr3 = np.abs(np.subtract(data_low[1:], data_close[:-1]))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])  # Align with close index
        # Sum of TR over period
        tr_sum = np.full_like(data_close, np.nan, dtype=float)
        for i in range(period, len(data_close)):
            tr_sum[i] = np.nansum(tr[i-period+1:i+1])
        # Highest high and lowest low over period
        hh = np.full_like(data_close, np.nan, dtype=float)
        ll = np.full_like(data_close, np.nan, dtype=float)
        for i in range(period-1, len(data_close)):
            hh[i] = np.max(data_high[i-period+1:i+1])
            ll[i] = np.min(data_low[i-period+1:i+1])
        # Chop calculation
        chop_vals = np.full_like(data_close, np.nan, dtype=float)
        for i in range(period-1, len(data_close)):
            if tr_sum[i] > 0 and (hh[i] - ll[i]) > 0:
                chop_vals[i] = 100 * np.log10(tr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
            else:
                chop_vals[i] = np.nan
        return chop_vals
    
    chop_vals = chop(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_vals)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: KAMA turns down OR RSI overbought OR chop becomes low (trending)
            if (close[i] < kama_vals[i]) or (rsi_vals[i] > 70) or (chop_aligned[i] < 38.2):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: KAMA turns up OR RSI oversold OR chop becomes low (trending)
            if (close[i] > kama_vals[i]) or (rsi_vals[i] < 30) or (chop_aligned[i] < 38.2):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Chop filter: only enter when market is ranging (chop > 61.8)
            if chop_aligned[i] > 61.8:
                # Uptrend: price above KAMA AND RSI not overbought
                if (close[i] > kama_vals[i]) and (rsi_vals[i] < 70):
                    position = 1
                    signals[i] = 0.25
                # Downtrend: price below KAMA AND RSI not oversold
                elif (close[i] < kama_vals[i]) and (rsi_vals[i] > 30):
                    position = -1
                    signals[i] = -0.25
    
    return signals