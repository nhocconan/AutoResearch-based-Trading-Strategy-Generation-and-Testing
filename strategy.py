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
    
    # Get 1-day data for ATR and close
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR on daily data
    tr1 = np.maximum(high_1d[1:], low_1d[:-1]) - np.minimum(high_1d[1:], low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    atr_14 = np.full(len(close_1d), np.nan)
    for i in range(14, len(tr)):
        atr_14[i] = np.nanmean(tr[i-13:i+1])
    
    # Align daily ATR to 4h timeframe
    atr_14_4h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 4-period RSI on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/4, adjust=False, min_periods=4).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/4, adjust=False, min_periods=4).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_4 = 100 - (100 / (1 + rs))
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20)  # need RSI and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_14_4h[i]) or np.isnan(rsi_4[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: RSI oversold (<30) with volume confirmation
            if rsi_4[i] < 30 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: RSI overbought (>70) with volume confirmation
            elif rsi_4[i] > 70 and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: RSI overbought (>70) or price drops 1.5 * ATR from entry
            if rsi_4[i] > 70 or close[i] < close[i-1] - 1.5 * atr_14_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold (<30) or price rises 1.5 * ATR from entry
            if rsi_4[i] < 30 or close[i] > close[i-1] + 1.5 * atr_14_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI4_Volume_ATR_Stop"
timeframe = "4h"
leverage = 1.0