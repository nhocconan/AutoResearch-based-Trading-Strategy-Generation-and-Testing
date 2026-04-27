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
    
    # Get daily data for 14-day ATR (volatility filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for daily ATR
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate ATR(14) for daily timeframe
    atr_1d = np.zeros(len(tr_1d))
    for i in range(len(tr_1d)):
        if i < 13:
            atr_1d[i] = np.mean(tr_1d[:i+1]) if i > 0 else tr_1d[i]
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Align daily ATR to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Get weekly data for EMA(34) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    alpha_w = 2 / (34 + 1)
    ema_1w_34 = np.full(len(df_1w), np.nan)
    for i in range(len(close_1w)):
        if i < 33:
            ema_1w_34[i] = np.mean(close_1w[:i+1]) if i > 0 else close_1w[i]
        else:
            if np.isnan(ema_1w_34[i-1]):
                ema_1w_34[i] = np.mean(close_1w[i-33:i+1])
            else:
                ema_1w_34[i] = close_1w[i] * alpha_w + ema_1w_34[i-1] * (1 - alpha_w)
    
    ema_1w_34_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34)
    
    # Calculate 4-period RSI for momentum (short-term)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(n):
        if i < 3:
            avg_gain[i] = np.mean(gain[:i+1]) if i > 0 else gain[i]
            avg_loss[i] = np.mean(loss[:i+1]) if i > 0 else loss[i]
        else:
            avg_gain[i] = (avg_gain[i-1] * 3 + gain[i]) / 4
            avg_loss[i] = (avg_loss[i-1] * 3 + loss[i]) / 4
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume moving average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(34, 20)  # weekly EMA needs 34, volume MA needs 20
    
    for i in range(start_idx, n):
        if (np.isnan(atr_1d_aligned[i]) or
            np.isnan(ema_1w_34_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volatility filter: only trade when volatility is elevated (ATR > 1.5x 20-period average)
        # Calculate 20-period ATR average for volatility regime
        if i >= 20 + 13:  # Need enough data for 20-period ATR average
            atr_20_avg = np.mean(atr_1d_aligned[i-20:i])
            vol_filter = atr_1d_aligned[i] > 1.5 * atr_20_avg
        else:
            vol_filter = False
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: RSI < 30 (oversold) + volatility filter + volume confirmation + weekly uptrend
            if (rsi[i] < 30 and 
                vol_filter and 
                volume_confirmation and 
                ema_1w_34_aligned[i] > ema_1w_34_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + volatility filter + volume confirmation + weekly downtrend
            elif (rsi[i] > 70 and 
                  vol_filter and 
                  volume_confirmation and 
                  ema_1w_34_aligned[i] < ema_1w_34_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI > 70 (overbought) or weekly trend turns down
            if (rsi[i] > 70 or 
                ema_1w_34_aligned[i] < ema_1w_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: RSI < 30 (oversold) or weekly trend turns up
            if (rsi[i] < 30 or 
                ema_1w_34_aligned[i] > ema_1w_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "4h_VolatilityFilter_RSI_WeeklyEMA34_v1"
timeframe = "4h"
leverage = 1.0