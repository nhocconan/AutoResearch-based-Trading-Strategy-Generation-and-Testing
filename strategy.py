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
    
    # Get 1d data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on daily timeframe
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        atr_14[i] = np.nanmean(tr[i-13:i+1])
    
    # Align ATR to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 4-period RSI on 4h close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    for i in range(14, len(close)):
        if i == 14:
            avg_gain[i] = np.mean(gain[0:14])
            avg_loss[i] = np.mean(loss[0:14])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 4-period SMA of RSI for smoothing
    rsi_sma = np.full_like(rsi, np.nan)
    for i in range(3, len(rsi)):
        rsi_sma[i] = np.mean(rsi[i-3:i+1])
    
    # Volume filter: volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need ATR (14), RSI (14+3), volume MA (20)
    start_idx = max(14+3, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_14_aligned[i]) or np.isnan(rsi_sma[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_14_aligned[i]
        rsi_val = rsi_sma[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volatility filter: only trade when ATR > 0.5% of price (avoid choppy markets)
        vol_filter = atr > 0.005 * price
        
        # Volume filter: moderate volume confirmation
        vol_conf = vol_now > 1.5 * vol_avg
        
        # RSI conditions: oversold (<30) for long, overbought (>70) for short
        rsi_oversold = rsi_val < 30
        rsi_overbought = rsi_val > 70
        
        if position == 0:
            # Long: RSI oversold + volatility + volume confirmation
            if rsi_oversold and vol_filter and vol_conf:
                signals[i] = size
                position = 1
            # Short: RSI overbought + volatility + volume confirmation
            elif rsi_overbought and vol_filter and vol_conf:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (50) or volatility drops
            if rsi_val >= 50 or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI returns to neutral (50) or volatility drops
            if rsi_val <= 50 or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_RSI_Volatility_Filter_Volume"
timeframe = "4h"
leverage = 1.0