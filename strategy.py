# 1d_KAMA_Trend_12h_TrendFilter_VolumeConfirmation_v1
# KAMA trend on 1d timeframe for primary trend direction
# Confirmed by 12h EMA(50) trend filter and volume (>1.5x 20-bar average)
# Entry when price crosses above/below KAMA with trend filter and volume confirmation
# Exit when price crosses back below/above KAMA
# KAMA adapts to market conditions - fast in trends, slow in ranges
# Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year)
# Works in both bull/bear: captures strong trends, avoids false signals in consolidation

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_Trend_12h_TrendFilter_VolumeConfirmation_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_1d) < 30 or len(df_12h) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate KAMA on 1d timeframe
    # Efficiency Ratio (ER) = |change| / volatility
    # Smoothing Constant (SC) = [ER * (fastest - slowest) + slowest]^2
    # KAMA = previous KAMA + SC * (price - previous KAMA)
    def kama(data, period=10, fast=2, slow=30):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        
        change = np.abs(np.diff(data, n=period))
        volatility = np.nansum(np.abs(np.diff(data)), axis=0) if len(data) > 1 else 0
        # For rolling window calculation
        er = np.full_like(data, np.nan)
        sc = np.full_like(data, np.nan)
        
        for i in range(period, len(data)):
            if i >= period:
                ch = np.abs(data[i] - data[i-period])
                vol = np.nansum(np.abs(data[i-period+1:i+1] - data[i-period:i]))
                if vol != 0:
                    er[i] = ch / vol
                else:
                    er[i] = 0
                sc[i] = (er[i] * (fast - slow) + slow) ** 2
                sc[i] = sc[i] ** 2  # Final smoothing constant
        
        # Initialize KAMA
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            if not np.isnan(sc[i]):
                result[i] = result[i-1] + sc[i] * (data[i] - result[i-1])
            else:
                result[i] = result[i-1]
        return result
    
    kama_1d = kama(close_1d, 10, 2, 30)
    
    # Calculate EMA(50) on 12h timeframe for trend filter
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False).values
    
    # Calculate volume confirmation filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 1d timeframe (primary)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(ema_12h_aligned[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price crosses above KAMA AND uptrend (price > EMA) AND volume confirmation
            if (close[i] > kama_1d_aligned[i] and close[i] > ema_12h_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price crosses below KAMA AND downtrend (price < EMA) AND volume confirmation
            elif (close[i] < kama_1d_aligned[i] and close[i] < ema_12h_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA
            if close[i] < kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above KAMA
            if close[i] > kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals