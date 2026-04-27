#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR and volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(20) for volatility regime
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_20_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(tr_1d)):
        atr_20_1d[i] = np.mean(tr_1d[i-20:i])
    
    atr_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_20_1d)
    
    # Calculate ATR ratio: current ATR(10) / ATR(20) - volatility expansion signal
    tr1_10 = high_1d[1:] - low_1d[1:]
    tr2_10 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_10 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_10d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1_10, np.maximum(tr2_10, tr3_10))])
    
    atr_10_1d = np.full(len(df_1d), np.nan)
    for i in range(10, len(tr_10d)):
        atr_10_1d[i] = np.mean(tr_10d[i-10:i])
    
    atr_10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_10_1d)
    
    # ATR ratio: ATR(10)/ATR(20) > 1.4 indicates volatility expansion
    atr_ratio = np.full(n, np.nan)
    valid_mask = (~np.isnan(atr_10_1d_aligned)) & (~np.isnan(atr_20_1d_aligned)) & (atr_20_1d_aligned > 0)
    atr_ratio[valid_mask] = atr_10_1d_aligned[valid_mask] / atr_20_1d_aligned[valid_mask]
    
    # Get weekly data for trend filter: EMA(34) on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w_34 = np.full(len(df_1w), np.nan)
    alpha_w = 2 / (34 + 1)
    for i in range(len(close_1w)):
        if i < 33:
            ema_1w_34[i] = np.mean(close_1w[:i+1]) if i > 0 else close_1w[i]
        else:
            if np.isnan(ema_1w_34[i-1]):
                ema_1w_34[i] = np.mean(close_1w[i-33:i+1])
            else:
                ema_1w_34[i] = close_1w[i] * alpha_w + ema_1w_34[i-1] * (1 - alpha_w)
    
    ema_1w_34_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34)
    
    # Calculate daily RSI(14) for mean reversion signals
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    
    avg_gain_1d = np.full(len(df_1d), np.nan)
    avg_loss_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(gain)):
        if i == 14:
            avg_gain_1d[i] = np.mean(gain[1:15])
            avg_loss_1d[i] = np.mean(loss[1:15])
        else:
            avg_gain_1d[i] = (avg_gain_1d[i-1] * 13 + gain[i]) / 14
            avg_loss_1d[i] = (avg_loss_1d[i-1] * 13 + loss[i]) / 14
    
    rs_1d = np.full(len(df_1d), np.nan)
    valid_rsi_1d = (~np.isnan(avg_gain_1d)) & (~np.isnan(avg_loss_1d)) & (avg_loss_1d > 0)
    rs_1d[valid_rsi_1d] = avg_gain_1d[valid_rsi_1d] / avg_loss_1d[valid_rsi_1d]
    rsi_14_1d = np.full(len(df_1d), np.nan)
    rsi_14_1d[valid_rsi_1d] = 100 - (100 / (1 + rs_1d[valid_rsi_1d]))
    
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate daily moving average for trend filter
    sma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(close_1d)):
        sma_20_1d[i] = np.mean(close_1d[i-20:i])
    
    sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_20_1d)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(20, 34, 14)
    
    for i in range(start_idx, n):
        if (np.isnan(atr_ratio[i]) or 
            np.isnan(ema_1w_34_aligned[i]) or
            np.isnan(rsi_14_1d_aligned[i]) or
            np.isnan(sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volatility regime filter: ATR ratio > 1.4 = expansion (favor trend)
        vol_expansion = atr_ratio[i] > 1.4
        
        if position == 0:
            # Long: RSI < 30 (oversold) + volatility expansion + price above SMA20 + weekly uptrend
            if (rsi_14_1d_aligned[i] < 30 and 
                vol_expansion and 
                price > sma_20_1d_aligned[i] and
                ema_1w_34_aligned[i] > ema_1w_34_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + volatility expansion + price below SMA20 + weekly downtrend
            elif (rsi_14_1d_aligned[i] > 70 and 
                  vol_expansion and 
                  price < sma_20_1d_aligned[i] and
                  ema_1w_34_aligned[i] < ema_1w_34_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI > 70 or weekly trend turns down or price below SMA20
            if (rsi_14_1d_aligned[i] > 70 or 
                ema_1w_34_aligned[i] < ema_1w_34_aligned[i-1] or
                price < sma_20_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 30 or weekly trend turns up or price above SMA20
            if (rsi_14_1d_aligned[i] < 30 or 
                ema_1w_34_aligned[i] > ema_1w_34_aligned[i-1] or
                price > sma_20_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_VolatilityExpansion_RSI14_SMA20_WeeklyEMA34_v1"
timeframe = "1d"
leverage = 1.0