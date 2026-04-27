#!/usr/bin/env python3
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
    volume = prices['volume'].values
    
    # Get daily data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14) for volatility regime
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(tr_1d)):
        atr_14_1d[i] = np.mean(tr_1d[i-14:i])
    
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate ATR ratio: current ATR(7) / ATR(14) - volatility expansion signal
    tr1_7 = high_1d[1:] - low_1d[1:]
    tr2_7 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_7 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_7d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1_7, np.maximum(tr2_7, tr3_7))])
    
    atr_7_1d = np.full(len(df_1d), np.nan)
    for i in range(7, len(tr_7d)):
        atr_7_1d[i] = np.mean(tr_7d[i-7:i])
    
    atr_7_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_7_1d)
    
    # ATR ratio: ATR(7)/ATR(14) > 1.3 indicates volatility expansion
    atr_ratio = np.full(n, np.nan)
    valid_mask = (~np.isnan(atr_7_1d_aligned)) & (~np.isnan(atr_14_1d_aligned)) & (atr_14_1d_aligned > 0)
    atr_ratio[valid_mask] = atr_7_1d_aligned[valid_mask] / atr_14_1d_aligned[valid_mask]
    
    # Calculate daily RSI(14) for mean reversion signals
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    
    avg_gain = np.full(len(df_1d), np.nan)
    avg_loss = np.full(len(df_1d), np.nan)
    for i in range(14, len(delta)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full(len(df_1d), np.nan)
    valid_rsi = (~np.isnan(avg_gain)) & (~np.isnan(avg_loss)) & (avg_loss > 0)
    rs[valid_rsi] = avg_gain[valid_rsi] / avg_loss[valid_rsi]
    rsi_14_1d = np.full(len(df_1d), np.nan)
    rsi_14_1d[valid_rsi] = 100 - (100 / (1 + rs[valid_rsi]))
    
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate daily Donchian channels (20-period)
    highest_high = np.full(len(df_1d), np.nan)
    lowest_low = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        highest_high[i] = np.max(high_1d[i-20:i])
        lowest_low[i] = np.min(low_1d[i-20:i])
    
    highest_high_aligned = align_htf_to_ltf(prices, df_1d, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_1d, lowest_low)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(20, 14, 7)
    
    for i in range(start_idx, n):
        if (np.isnan(atr_ratio[i]) or 
            np.isnan(rsi_14_1d_aligned[i]) or
            np.isnan(highest_high_aligned[i]) or
            np.isnan(lowest_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volatility regime filter: ATR ratio > 1.3 = expansion (favor breakout)
        vol_expansion = atr_ratio[i] > 1.3
        
        if position == 0:
            # Long: Price breaks above Donchian high + RSI < 70 (not overbought) + volatility expansion
            if (price > highest_high_aligned[i] and 
                rsi_14_1d_aligned[i] < 70 and 
                vol_expansion):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + RSI > 30 (not oversold) + volatility expansion
            elif (price < lowest_low_aligned[i] and 
                  rsi_14_1d_aligned[i] > 30 and 
                  vol_expansion):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price breaks below Donchian low or RSI > 70 (overbought)
            if (price < lowest_low_aligned[i] or 
                rsi_14_1d_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above Donchian high or RSI < 30 (oversold)
            if (price > highest_high_aligned[i] or 
                rsi_14_1d_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_RSI14_VolatilityExpansion_v1"
timeframe = "4h"
leverage = 1.0