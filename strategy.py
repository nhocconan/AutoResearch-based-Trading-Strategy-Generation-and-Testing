#!/usr/bin/env python3
"""
1d_1w_12h_RangeReversal_VolumeConfirm
Hypothesis: In the daily timeframe, price reversals from weekly Bollinger Bands (20,2) combined with 12h RSI extremes and volume spikes provide high-probability mean-reversion trades in both bull and bear markets. The weekly Bollinger Bands define the macro range, 12h RSI <30 or >70 signals exhaustion, and volume >2x average confirms participation. Designed for low trade frequency (target: 10-20/year) to minimize fee drag in 1d timeframe. Uses discrete position sizing (0.25).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Bollinger Bands (20,2)
    close_1w = df_1w['close'].values
    sma20_1w = np.full_like(close_1w, np.nan)
    std20_1w = np.full_like(close_1w, np.nan)
    for i in range(len(close_1w)):
        if i >= 19:
            sma20_1w[i] = np.mean(close_1w[i-19:i+1])
            std20_1w[i] = np.std(close_1w[i-19:i+1])
    
    upper_bb = sma20_1w + 2 * std20_1w
    lower_bb = sma20_1w - 2 * std20_1w
    
    # Align weekly Bollinger Bands to daily
    upper_bb_aligned = align_htf_to_ltf(prices, df_1w, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1w, lower_bb)
    
    # Load 12h data for RSI
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate 12h RSI(14)
    close_12h = df_12h['close'].values
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_12h, np.nan)
    avg_loss = np.full_like(close_12h, np.nan)
    for i in range(len(close_12h)):
        if i < 14:
            if i > 0:
                avg_gain[i] = np.mean(gain[1:i+1])
                avg_loss[i] = np.mean(loss[1:i+1])
            else:
                avg_gain[i] = gain[0]
                avg_loss[i] = loss[0]
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Daily data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 2x 20-period average
    volume_avg = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 20:
            volume_avg[i] = np.mean(volume[i-20:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (2 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or np.isnan(rsi_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ok = volume_filter[i]
        upper_bb_val = upper_bb_aligned[i]
        lower_bb_val = lower_bb_aligned[i]
        rsi_val = rsi_12h_aligned[i]
        
        if position == 0:
            # Long: price touches or breaks below lower weekly BB, RSI oversold, volume spike
            if price <= lower_bb_val and rsi_val < 30 and vol_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price touches or breaks above upper weekly BB, RSI overbought, volume spike
            elif price >= upper_bb_val and rsi_val > 70 and vol_ok:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price returns to weekly SMA20 or RSI normalizes
            if price >= sma20_1w[i] if not np.isnan(sma20_1w[i]) else False or rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to weekly SMA20 or RSI normalizes
            if price <= sma20_1w[i] if not np.isnan(sma20_1w[i]) else False or rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_12h_RangeReversal_VolumeConfirm"
timeframe = "1d"
leverage = 1.0