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
    
    # Get daily data for ATR and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(tr_1d)):
        atr_14_1d[i] = np.mean(tr_1d[i-14:i])
    
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
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
    
    # Calculate volume spike: current volume > 2 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    volume_spike = np.full(n, False)
    vol_valid = ~np.isnan(vol_ma) & (vol_ma > 0)
    volume_spike[vol_valid] = volume[vol_valid] > (2 * vol_ma[vol_valid])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(30, 4, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(rsi_4[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volatility filter: ATR > 0.02 * price (avoid low volatility chop)
        vol_filter = atr_14_1d_aligned[i] > (0.02 * price)
        
        if position == 0:
            # Long: RSI < 25 (oversold) + volume spike + volatility filter
            if (rsi_4[i] < 25 and 
                volume_spike[i] and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 75 (overbought) + volume spike + volatility filter
            elif (rsi_4[i] > 75 and 
                  volume_spike[i] and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI > 60 or volatility drops
            if (rsi_4[i] > 60 or 
                not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 40 or volatility drops
            if (rsi_4[i] < 40 or 
                not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_VolumeSpike_RSI4_ATRFilter_v1"
timeframe = "4h"
leverage = 1.0