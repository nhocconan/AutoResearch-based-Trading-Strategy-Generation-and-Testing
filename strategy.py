#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 14-period RSI
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    delta = np.insert(delta, 0, 0)
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
    
    # Calculate 10-period EMA
    ema_10_1d = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_10_1d)
    
    # Calculate 20-period EMA
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate 20-period ATR for position sizing and volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Wilder's smoothing for ATR
    atr_1d = wilder_smooth(tr, 20)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Session filter: 8-20 UTC
    hours = prices.index.hour  # Pre-compute before loop
    
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
        ema_10_val = ema_10_1d_aligned[i]
        ema_20_val = ema_20_1d_aligned[i]
        atr_val = atr_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(ema_10_val) or 
            np.isnan(ema_20_val) or np.isnan(atr_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) and EMA10 > EMA20 (bullish momentum)
            if rsi_val < 30 and ema_10_val > ema_20_val:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) and EMA10 < EMA20 (bearish momentum)
            elif rsi_val > 70 and ema_10_val < ema_20_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 70 (overbought) or EMA10 < EMA20 (momentum lost)
            if rsi_val > 70 or ema_10_val < ema_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI < 30 (oversold) or EMA10 > EMA20 (momentum lost)
            if rsi_val < 30 or ema_10_val > ema_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 6h_RSI_EMA_Momentum_Reversal
# Uses daily RSI(14) for overbought/oversold signals
# Uses daily EMA(10)/EMA(20) crossover for momentum confirmation
# Entry: RSI < 30 + EMA10 > EMA20 (long) or RSI > 70 + EMA10 < EMA20 (short)
# Exit: RSI > 70 or EMA10 < EMA20 (long); RSI < 30 or EMA10 > EMA20 (short)
# Session filter: 8-20 UTC to avoid low-volume periods
# Position size: 0.25 (25% of capital)
name = "6h_RSI_EMA_Momentum_Reversal"
timeframe = "6h"
leverage = 1.0