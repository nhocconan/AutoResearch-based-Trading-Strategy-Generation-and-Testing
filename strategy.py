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
    
    # === 1d VWAP (Volume Weighted Average Price) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate typical price and VWAP components
    tp_1d = (high_1d + low_1d + close_1d) / 3
    tpv_1d = tp_1d * volume_1d
    
    # Cumulative sums for VWAP
    cum_tpv = np.cumsum(tpv_1d)
    cum_vol = np.cumsum(volume_1d)
    vwap_1d = np.divide(cum_tpv, cum_vol, out=np.full_like(cum_tpv, np.nan), where=cum_vol!=0)
    
    # === 1d RSI (14-period) ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i < 14:
            if i > 0:
                avg_gain[i] = np.mean(gain[1:i+1])
                avg_loss[i] = np.mean(loss[1:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # === Align indicators to 1h timeframe ===
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 1h Volume confirmation ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    vol_confirm = volume > vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above VWAP AND RSI oversold (<30)
            if close[i] > vwap_aligned[i] and rsi_aligned[i] < 30:
                if vol_confirm[i]:
                    signals[i] = 0.20
                    position = 1
                    continue
            # Short: price below VWAP AND RSI overbought (>70)
            elif close[i] < vwap_aligned[i] and rsi_aligned[i] > 70:
                if vol_confirm[i]:
                    signals[i] = -0.20
                    position = -1
                    continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price below VWAP OR RSI overbought (>70)
            if close[i] < vwap_aligned[i] or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price above VWAP OR RSI oversold (<30)
            if close[i] > vwap_aligned[i] or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_VWAP_RSI_Volume_Confluence_v1"
timeframe = "1h"
leverage = 1.0