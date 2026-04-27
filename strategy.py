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
    
    # Get daily data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
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
    
    # Calculate daily RSI(14)
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.maximum(delta_1d, 0)
    loss_1d = np.maximum(-delta_1d, 0)
    
    avg_gain_1d = np.full(len(df_1d), np.nan)
    avg_loss_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(close_1d)):
        if i == 14:
            avg_gain_1d[i] = np.mean(gain_1d[1:15])
            avg_loss_1d[i] = np.mean(loss_1d[1:15])
        else:
            avg_gain_1d[i] = (avg_gain_1d[i-1] * 13 + gain_1d[i]) / 14
            avg_loss_1d[i] = (avg_loss_1d[i-1] * 13 + loss_1d[i]) / 14
    
    rs_1d = np.full(len(df_1d), np.nan)
    valid_rsi_1d = (~np.isnan(avg_gain_1d)) & (~np.isnan(avg_loss_1d)) & (avg_loss_1d > 0)
    rs_1d[valid_rsi_1d] = avg_gain_1d[valid_rsi_1d] / avg_loss_1d[valid_rsi_1d]
    rsi_14_1d = np.full(len(df_1d), np.nan)
    rsi_14_1d[valid_rsi_1d] = 100 - (100 / (1 + rs_1d[valid_rsi_1d]))
    
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate daily volume ratio: current volume / 20-day average volume
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-20:i])
    
    vol_ratio_1d = np.full(len(df_1d), np.nan)
    valid_vol = (~np.isnan(volume_1d)) & (~np.isnan(vol_ma_20_1d)) & (vol_ma_20_1d > 0)
    vol_ratio_1d[valid_vol] = volume_1d[valid_vol] / vol_ma_20_1d[valid_vol]
    
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(34, 14, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(rsi_14_1d_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volatility filter: ATR > 0.02 * price (avoid low volatility chop)
        vol_filter = atr_14_1d_aligned[i] > 0.02 * price
        
        # Volume filter: volume ratio > 1.5 (above average volume)
        vol_spike = vol_ratio_1d_aligned[i] > 1.5
        
        if position == 0:
            # Long: RSI < 40 (oversold) + volatility + volume spike
            if (rsi_14_1d_aligned[i] < 40 and 
                vol_filter and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 60 (overbought) + volatility + volume spike
            elif (rsi_14_1d_aligned[i] > 60 and 
                  vol_filter and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI > 60 or volatility drops
            if (rsi_14_1d_aligned[i] > 60 or 
                not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 40 or volatility drops
            if (rsi_14_1d_aligned[i] < 40 or 
                not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_VolumeSpike_RSI14_VolatilityFilter_v1"
timeframe = "4h"
leverage = 1.0