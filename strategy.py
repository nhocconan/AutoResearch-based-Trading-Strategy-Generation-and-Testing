#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 14-period RSI for 12h
    close_12h = df_12h['close'].values
    delta = np.diff(close_12h)
    delta = np.concatenate([[0], delta])
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
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Calculate 12h ATR for volatility filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_12h = wilder_smooth(tr, 14)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate 12h EMA20 for trend filter
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Calculate 60-period average volume (6h * 10 = 60h ≈ 2.5 days)
    volume_6h = prices['volume'].values
    vol_avg_60 = pd.Series(volume_6h).rolling(window=60, min_periods=60).mean().values
    
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
        rsi_val = rsi_12h_aligned[i]
        ema20_val = ema20_12h_aligned[i]
        atr_val = atr_12h_aligned[i]
        vol_val = volume_6h[i]
        vol_avg_val = vol_avg_60[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(ema20_val) or 
            np.isnan(atr_val) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold), price above EMA20, volume above average
            if rsi_val < 30 and close_val > ema20_val and vol_val > vol_avg_val:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought), price below EMA20, volume above average
            elif rsi_val > 70 and close_val < ema20_val and vol_val > vol_avg_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 70 (overbought) or price below EMA20
            if rsi_val > 70 or close_val < ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI < 30 (oversold) or price above EMA20
            if rsi_val < 30 or close_val > ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 6h_12h_RSI_MeanReversion_EMAFilter_Volume_Session_v1
# Uses 12h RSI for mean reversion signals (RSI < 30 long, > 70 short)
# Requires price to be on correct side of 12h EMA20 for trend alignment
# Volume confirmation: current volume > 60-period average
# Session filter: 8-20 UTC to avoid low-volume periods
# Designed for 6h timeframe with ~15-25 trades/year
name = "6h_12h_RSI_MeanReversion_EMAFilter_Volume_Session_v1"
timeframe = "6h"
leverage = 1.0