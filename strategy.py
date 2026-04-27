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
    
    # Get daily data for volatility regime and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
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
    
    # Calculate daily EMA(34) for trend filter
    ema_34_1d = np.full(len(df_1d), np.nan)
    alpha = 2 / (34 + 1)
    for i in range(len(close_1d)):
        if i < 33:
            ema_34_1d[i] = np.mean(close_1d[:i+1]) if i > 0 else close_1d[i]
        else:
            if np.isnan(ema_34_1d[i-1]):
                ema_34_1d[i] = np.mean(close_1d[i-33:i+1])
            else:
                ema_34_1d[i] = close_1d[i] * alpha + ema_34_1d[i-1] * (1 - alpha)
    
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12-period RSI for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(12, n):
        if i == 12:
            avg_gain[i] = np.mean(gain[1:13])
            avg_loss[i] = np.mean(loss[1:13])
        else:
            avg_gain[i] = (avg_gain[i-1] * 11 + gain[i]) / 12
            avg_loss[i] = (avg_loss[i-1] * 11 + loss[i]) / 12
    
    rs = np.full(n, np.nan)
    valid_rsi = (~np.isnan(avg_gain)) & (~np.isnan(avg_loss)) & (avg_loss > 0)
    rs[valid_rsi] = avg_gain[valid_rsi] / avg_loss[valid_rsi]
    rsi_12 = np.full(n, np.nan)
    rsi_12[valid_rsi] = 100 - (100 / (1 + rs[valid_rsi]))
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(14, 34, 12)
    
    for i in range(start_idx, n):
        if (np.isnan(atr_ratio[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(rsi_12[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volatility regime filter: ATR ratio > 1.3 = expansion (favor trend)
        vol_expansion = atr_ratio[i] > 1.3
        
        if position == 0:
            # Long: RSI < 30 (oversold) + volatility expansion + daily uptrend
            if (rsi_12[i] < 30 and 
                vol_expansion and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + volatility expansion + daily downtrend
            elif (rsi_12[i] > 70 and 
                  vol_expansion and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI > 70 or daily trend turns down
            if (rsi_12[i] > 70 or 
                ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 30 or daily trend turns up
            if (rsi_12[i] < 30 or 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_VolatilityExpansion_RSI12_DailyEMA34_v1"
timeframe = "12h"
leverage = 1.0