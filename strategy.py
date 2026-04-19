# 1d_KAMA_With_RSI_And_Chop_Filter
# Hypothesis: KAMA adapts to market volatility, reducing whipsaw in ranging markets. 
# Combined with RSI overbought/oversold and Choppiness Index regime filter to avoid false signals.
# Works in bull via trend-following KAMA, in bear via mean-reversion when chop high.
# Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year).

name = "1d_KAMA_With_RSI_And_Chop_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAUFMAN ADAPTIVE MOVING AVERAGE (KAMA)
    def kama(price, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(price, n=period))
        volatility = np.sum(np.abs(np.diff(price)), axis=1)
        er = np.zeros_like(price)
        er[period:] = change[period-1:] / volatility[period-1:]
        er[np.isnan(er)] = 0
        er[er > 1] = 1
        
        # Smoothing Constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA calculation
        kama_val = np.zeros_like(price)
        kama_val[0] = price[0]
        for i in range(1, len(price)):
            if not np.isnan(sc[i]):
                kama_val[i] = kama_val[i-1] + sc[i] * (price[i] - kama_val[i-1])
            else:
                kama_val[i] = kama_val[i-1]
        return kama_val
    
    # RSI
    def rsi(price, period=14):
        delta = np.diff(price)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(price)
        avg_loss = np.zeros_like(price)
        
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period+1, len(price)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val
    
    # CHOPPINESS INDEX
    def choppiness_index(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Sum of True Range over period
        atr_sum = np.zeros_like(close)
        for i in range(period, len(close)):
            atr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        hh = np.zeros_like(close)
        ll = np.zeros_like(close)
        for i in range(period-1, len(close)):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        
        # Chop calculation
        chop = np.zeros_like(close)
        for i in range(period-1, len(close)):
            if hh[i] != ll[i]:
                log_sum = np.log10(atr_sum[i] / (hh[i] - ll[i]))
                chop[i] = 100 * log_sum / np.log10(period)
            else:
                chop[i] = 50  # neutral when no range
        return chop
    
    # 1d data for all indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate indicators on 1d data
    kama_val = kama(df_1d['close'].values, 10, 2, 30)
    rsi_val = rsi(df_1d['close'].values, 14)
    chop_val = choppiness_index(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Align to 1d timeframe (no additional delay needed as these are contemporaneous indicators)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_val)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_val)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_val)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: Chop > 61.8 = ranging (mean revert), Chop < 38.2 = trending (trend follow)
        is_ranging = chop_aligned[i] > 61.8
        is_trending = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long conditions
            long_signal = False
            if is_trending:
                # Trend following: price above KAMA
                long_signal = close[i] > kama_aligned[i]
            elif is_ranging:
                # Mean reversion: RSI oversold
                long_signal = rsi_aligned[i] < 30
            
            # Short conditions
            short_signal = False
            if is_trending:
                # Trend following: price below KAMA
                short_signal = close[i] < kama_aligned[i]
            elif is_ranging:
                # Mean reversion: RSI overbought
                short_signal = rsi_aligned[i] > 70
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long exit conditions
            exit_signal = False
            if is_trending:
                # Exit trend: price below KAMA
                exit_signal = close[i] < kama_aligned[i]
            elif is_ranging:
                # Exit mean reversion: RSI neutral or overbought
                exit_signal = rsi_aligned[i] > 50
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit conditions
            exit_signal = False
            if is_trending:
                # Exit trend: price above KAMA
                exit_signal = close[i] > kama_aligned[i]
            elif is_ranging:
                # Exit mean reversion: RSI neutral or oversold
                exit_signal = rsi_aligned[i] < 50
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals