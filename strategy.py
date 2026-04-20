#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly ATR for volatility normalization
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for weekly
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w[0] = tr1[0]
    
    # Weekly ATR (14-period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1w = wilder_smooth(tr_1w, 14)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate daily RSI (14-period)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = wilder_smooth(gain, 14)
    avg_loss = wilder_smooth(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate daily ATR for position sizing
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1_d = high_1d - low_1d
    tr2_d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    tr_1d[0] = tr1_d[0]
    
    atr_1d = wilder_smooth(tr_1d, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
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
        atr_1w_val = atr_1w_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(atr_1w_val) or np.isnan(atr_1d_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when weekly ATR is above its 20-period average
        # Calculate 20-period average of weekly ATR
        if i >= 100:  # Ensure we have enough history
            atr_1w_window = atr_1w_aligned[max(0, i-19):i+1]
            atr_1w_avg = np.mean(atr_1w_window) if len(atr_1w_window) > 0 else 0
            vol_filter = atr_1w_val > atr_1w_avg * 0.8  # Trade when volatility is not too low
        else:
            vol_filter = True
        
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
            # Long exit: RSI > 50 (mean reversion) or volatility drops significantly
            if rsi_val > 50 or (atr_1w_val < atr_1w_avg * 0.5 if 'atr_1w_avg' in locals() else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI < 50 (mean reversion) or volatility drops significantly
            if rsi_val < 50 or (atr_1w_val < atr_1w_avg * 0.5 if 'atr_1w_avg' in locals() else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 6h_WeeklyATR_DailyRSI_MeanReversion_VolatilityFilter_v1
# Uses weekly ATR for volatility regime filtering
# Uses daily RSI for mean reversion signals
# Long when RSI < 30 (oversold), Short when RSI > 70 (overbought)
# Only trades when volatility is sufficient (weekly ATR > 80% of its 20-period average)
# Exits when RSI mean reverts (>50 for longs, <50 for shorts) or volatility drops
# Session filter: 8-20 UTC to avoid low-volume periods
# Designed for 6h timeframe with ~15-30 trades/year
name = "6h_WeeklyATR_DailyRSI_MeanReversion_VolatilityFilter_v1"
timeframe = "6h"
leverage = 1.0