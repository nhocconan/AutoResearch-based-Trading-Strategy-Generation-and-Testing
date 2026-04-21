#!/usr/bin/env python3
"""
4h_1d_RSI_Confluence_VolumeBreakout
Hypothesis: RSI(14) crossing above 60 or below 40 with volume confirmation (>1.5x 20-bar average) and aligned daily trend (price > daily EMA50 for longs, < for shorts) captures momentum bursts. Works in bull/bear by requiring trend alignment. Uses ATR-based stoploss. Target ~30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h data for RSI and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14) with Wilder's smoothing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: volume > 1.5x 20-bar average
    volume_avg = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            volume_avg[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            volume_avg[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.5 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(rsi[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        ema50 = ema50_1d_aligned[i]
        vol_confirm = volume_filter[i]
        
        # Calculate ATR for stoploss (14-period)
        if i >= 14:
            tr_values = []
            for j in range(1, 15):
                idx = i - j
                if idx >= 0:
                    tr = max(high[idx] - low[idx], abs(high[idx] - close[idx-1]), abs(low[idx] - close[idx-1]))
                    tr_values.append(tr)
            atr = np.mean(tr_values) if tr_values else 0
        else:
            atr = 0
        
        # Stoploss: 2.0 * ATR from entry
        if position == 1 and price < entry_price - 2.0 * atr:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and price > entry_price + 2.0 * atr:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: RSI > 60 with volume confirmation in uptrend (price > daily EMA50)
            if rsi_val > 60 and vol_confirm and price > ema50:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: RSI < 40 with volume confirmation in downtrend (price < daily EMA50)
            elif rsi_val < 40 and vol_confirm and price < ema50:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: RSI < 50 or trend breaks
            if rsi_val < 50 or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI > 50 or trend breaks
            if rsi_val > 50 or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_RSI_Confluence_VolumeBreakout"
timeframe = "4h"
leverage = 1.0