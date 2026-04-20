#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d < 50):
        return np.zeros(n)
    
    # Calculate daily RSI(14) for momentum
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    avg_gain = wilder_smooth(gain, 14)
    avg_loss = wilder_smooth(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_14 = wilder_smooth(tr, 14)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate daily range (high-low)
    daily_range = high_1d - low_1d
    range_ma_10 = pd.Series(daily_range).rolling(window=10, min_periods=10).mean().values
    range_ma_10_aligned = align_htf_to_ltf(prices, df_1d, range_ma_10)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = prices['close'].iloc[i]
        rsi_val = rsi_1d_aligned[i]
        atr_val = atr_14_aligned[i]
        range_ma_val = range_ma_10_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(atr_val) or np.isnan(range_ma_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when daily range > 10-day average range
        vol_filter = daily_range[i] > range_ma_val if i < len(daily_range) else False
        
        if position == 0:
            # Long: RSI < 30 (oversold) and volatility filter
            if rsi_val < 30 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) and volatility filter
            elif rsi_val > 70 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 50 (mean reversion) or volatility filter fails
            if rsi_val > 50 or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI < 50 (mean reversion) or volatility filter fails
            if rsi_val < 50 or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 4h_DailyRSI_MeanReversion_VolFilter_Session_v1
# Uses daily RSI(14) for mean reversion signals
# Enters long when RSI < 30 (oversold), short when RSI > 70 (overbought)
# Filters by volatility: only trade when daily range > 10-day average range
# Uses daily ATR for volatility calculation
# Session filter: 8-20 UTC to avoid low-volume periods
# Exits when RSI crosses 50 (mean reversion) or volatility filter fails
# Designed for 4h timeframe with ~20-40 trades/year
name = "4h_DailyRSI_MeanReversion_VolFilter_Session_v1"
timeframe = "4h"
leverage = 1.0