#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_RMAScaled_Trend_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA21 for trend direction
    ema21_1d = pd.Series(df_1d['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 1d ATR(14) for volatility filter
    high_low = df_1d['high'] - df_1d['low']
    high_close = np.abs(df_1d['high'] - df_1d['close'].shift())
    low_close = np.abs(df_1d['low'] - df_1d['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 6h
    ema21_1d_6h = align_htf_to_ltf(prices, df_1d, ema21_1d)
    atr14_1d_6h = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Need enough data for ATR and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema21_1d_6h[i]) or np.isnan(atr14_1d_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema21_1d_6h[i]
        atr = atr14_1d_6h[i]
        vol_filter = atr > 0  # Always true if ATR calculated
        
        # 6-period RMA of close (equivalent to Wilder's smoothing)
        if i >= 6:
            # Calculate RMA manually for current window
            rma_sum = 0.0
            alpha = 1.0 / 6
            for j in range(6):
                idx = i - j
                if j == 0:
                    rma_sum = close[idx]
                else:
                    rma_sum = alpha * close[idx] + (1 - alpha) * rma_sum
            rma_val = rma_sum
        else:
            rma_val = close[i]  # Fallback for early bars
        
        if position == 0:
            # Enter long: price > RMA and above trend
            if close[i] > rma_val and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: price < RMA and below trend
            elif close[i] < rma_val and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below RMA
            if close[i] < rma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above RMA
            if close[i] > rma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals