#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for 1-day close and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily ATR(14)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(tr_1d)):
        atr_14_1d[i] = np.mean(tr_1d[i-14:i])
    
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Get weekly data for Donchian channel
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian(20)
    donch_high_20_1w = np.full(len(df_1w), np.nan)
    donch_low_20_1w = np.full(len(df_1w), np.nan)
    for i in range(19, len(df_1w)):
        donch_high_20_1w[i] = np.max(high_1w[i-19:i+1])
        donch_low_20_1w[i] = np.min(low_1w[i-19:i+1])
    
    donch_high_20_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20_1w)
    donch_low_20_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20_1w)
    
    # Calculate 6-period RSI for mean reversion
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(6, n):
        if i == 6:
            avg_gain[i] = np.mean(gain[1:7])
            avg_loss[i] = np.mean(loss[1:7])
        else:
            avg_gain[i] = (avg_gain[i-1] * 5 + gain[i]) / 6
            avg_loss[i] = (avg_loss[i-1] * 5 + loss[i]) / 6
    
    rs = np.full(n, np.nan)
    valid_rsi = (~np.isnan(avg_gain)) & (~np.isnan(avg_loss)) & (avg_loss > 0)
    rs[valid_rsi] = avg_gain[valid_rsi] / avg_loss[valid_rsi]
    rsi_6 = np.full(n, np.nan)
    rsi_6[valid_rsi] = 100 - (100 / (1 + rs[valid_rsi]))
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(20, 20, 6)
    
    for i in range(start_idx, n):
        if (np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(donch_high_20_1w_aligned[i]) or
            np.isnan(donch_low_20_1w_aligned[i]) or
            np.isnan(rsi_6[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volatility filter: ATR > 0.5 * price (avoid low volatility chop)
        vol_filter = atr_14_1d_aligned[i] > 0.005 * price
        
        if position == 0:
            # Long: Price touches weekly Donchian low + RSI oversold + volatility filter
            if (price <= donch_low_20_1w_aligned[i] * 1.001 and  # Allow small slippage
                rsi_6[i] < 30 and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price touches weekly Donchian high + RSI overbought + volatility filter
            elif (price >= donch_high_20_1w_aligned[i] * 0.999 and  # Allow small slippage
                  rsi_6[i] > 70 and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI > 70 or price touches weekly Donchian high
            if (rsi_6[i] > 70 or 
                price >= donch_high_20_1w_aligned[i] * 0.999):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 30 or price touches weekly Donchian low
            if (rsi_6[i] < 30 or 
                price <= donch_low_20_1w_aligned[i] * 1.001):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyDonchian_RSI6_VolatilityFilter_v1"
timeframe = "6h"
leverage = 1.0