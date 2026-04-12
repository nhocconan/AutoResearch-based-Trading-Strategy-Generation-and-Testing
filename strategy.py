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
    
    # Get daily data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily RSI(14) for momentum
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        if i == 14:
            avg_gain[i] = np.mean(gain[0:15])
            avg_loss[i] = np.mean(loss[0:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr[i-14:i+1])
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly SMA(10) for trend filter
    sma_10_1w = np.full(len(close_1w), np.nan)
    for i in range(9, len(close_1w)):
        sma_10_1w[i] = np.mean(close_1w[i-9:i+1])
    
    # Align daily indicators to 4h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Align weekly SMA to 4h timeframe
    sma_10_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_10_1w)
    
    # Calculate 4h ATR(14) for position sizing
    tr1_h = np.abs(high - low)
    tr2_h = np.abs(high - np.roll(close, 1))
    tr3_h = np.abs(low - np.roll(close, 1))
    tr1_h[0] = tr2_h[0] = tr3_h[0] = np.nan
    tr_h = np.maximum(tr1_h, np.maximum(tr2_h, tr3_h))
    atr_4h = np.full(n, np.nan)
    for i in range(14, n):
        atr_4h[i] = np.mean(tr_h[i-14:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(sma_10_1w_aligned[i]) or np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: daily ATR > 0.5 * 20-period ATR MA (avoid low volatility)
        atr_ma_20 = np.full(n, np.nan)
        for j in range(33, n):  # 14 + 19 for 20-period MA
            if not np.isnan(np.mean(atr_1d_aligned[j-19:j+1])):
                atr_ma_20[j] = np.mean(atr_1d_aligned[j-19:j+1])
        vol_filter = atr_1d_aligned[i] > 0.5 * atr_ma_20[i] if not np.isnan(atr_ma_20[i]) else False
        
        # Momentum filter: daily RSI between 35 and 65 (avoid extremes)
        mom_filter = (rsi_1d_aligned[i] >= 35) and (rsi_1d_aligned[i] <= 65)
        
        # Trend filter: price above/both weekly SMA10
        uptrend = close[i] > sma_10_1w_aligned[i]
        downtrend = close[i] < sma_10_1w_aligned[i]
        
        # Entry conditions: RSI mean reversion with trend filter
        long_entry = (rsi_1d_aligned[i] < 40) and vol_filter and mom_filter and uptrend
        short_entry = (rsi_1d_aligned[i] > 60) and vol_filter and mom_filter and downtrend
        
        # Exit conditions: RSI crosses back to 50
        long_exit = rsi_1d_aligned[i] > 50
        short_exit = rsi_1d_aligned[i] < 50
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_1w_rsi_trend_filter_v1"
timeframe = "4h"
leverage = 1.0