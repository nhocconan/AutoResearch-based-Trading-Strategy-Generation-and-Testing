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
    
    # Get daily data for multiple indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR(14)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(tr_1d)):
        atr_14_1d[i] = np.mean(tr_1d[i-14:i])
    
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily ADX(14) for trend strength
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    tr_14 = np.zeros(len(tr_1d))
    tr_14[0] = tr_1d[0]
    for i in range(1, len(tr_1d)):
        tr_14[i] = tr_14[i-1] - (tr_14[i-1]/14) + tr_1d[i]
    
    plus_dm_14 = np.zeros(len(plus_dm))
    minus_dm_14 = np.zeros(len(minus_dm))
    for i in range(1, len(plus_dm)):
        plus_dm_14[i] = plus_dm_14[i-1] - (plus_dm_14[i-1]/14) + plus_dm[i]
        minus_dm_14[i] = minus_dm_14[i-1] - (minus_dm_14[i-1]/14) + minus_dm[i]
    
    plus_di = np.full(len(plus_dm_14), np.nan)
    minus_di = np.full(len(minus_dm_14), np.nan)
    valid_di = tr_14[14:] > 0
    if np.any(valid_di):
        plus_di[14:] = np.where(valid_di, 100 * plus_dm_14[14:] / tr_14[14:], 0)
        minus_di[14:] = np.where(valid_di, 100 * minus_dm_14[14:] / tr_14[14:], 0)
    
    dx = np.full(len(plus_di), np.nan)
    di_sum = plus_di + minus_di
    valid_dx = (di_sum > 0) & (~np.isnan(plus_di)) & (~np.isnan(minus_di))
    dx[valid_dx] = 100 * np.abs(plus_di[valid_dx] - minus_di[valid_dx]) / di_sum[valid_dx]
    
    adx_14 = np.full(len(dx), np.nan)
    for i in range(14, len(dx)):
        if not np.isnan(dx[i-1]):
            adx_14[i] = (adx_14[i-1] * 13 + dx[i]) / 14
        else:
            adx_14[i] = np.mean(dx[max(0, i-13):i+1])
    
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate 4-period RSI for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(4, n):
        if i == 4:
            avg_gain[i] = np.mean(gain[1:5])
            avg_loss[i] = np.mean(loss[1:5])
        else:
            avg_gain[i] = (avg_gain[i-1] * 3 + gain[i]) / 4
            avg_loss[i] = (avg_loss[i-1] * 3 + loss[i]) / 4
    
    rs = np.full(n, np.nan)
    valid_rsi = (~np.isnan(avg_gain)) & (~np.isnan(avg_loss)) & (avg_loss > 0)
    rs[valid_rsi] = avg_gain[valid_rsi] / avg_loss[valid_rsi]
    rsi_4 = np.full(n, np.nan)
    rsi_4[valid_rsi] = 100 - (100 / (1 + rs[valid_rsi]))
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(14, 4)
    
    for i in range(start_idx, n):
        if (np.isnan(adx_14_aligned[i]) or 
            np.isnan(rsi_4[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        is_trending = adx_14_aligned[i] > 25
        
        if position == 0:
            # Long: RSI < 30 (oversold) + trending market
            if (rsi_4[i] < 30 and 
                is_trending):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + trending market
            elif (rsi_4[i] > 70 and 
                  is_trending):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI > 70 or trend weakens
            if (rsi_4[i] > 70 or 
                adx_14_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 30 or trend weakens
            if (rsi_4[i] < 30 or 
                adx_14_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ADX25_RSI4_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0