#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    """
    4h Williams %R Reversal with weekly EMA trend and volume confirmation.
    Long when: Williams %R(14) < -80 (oversold) + price > weekly EMA50 + volume spike
    Short when: Williams %R(14) > -20 (overbought) + price < weekly EMA50 + volume spike
    Exits on opposite Williams %R cross or trailing stop.
    """
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * 2 + ema_50_1w[i-1] * 48) / 50  # EMA50
    
    # Calculate Williams %R (14) on 4h data
    williams_r = np.full(n, np.nan)
    lookback = 14
    for i in range(lookback - 1, n):
        highest_high = np.max(high[i - lookback + 1:i + 1])
        lowest_low = np.min(low[i - lookback + 1:i + 1])
        if highest_high != lowest_low:
            williams_r[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
        else:
            williams_r[i] = -50  # neutral if no range
    
    # Align weekly EMA50 to 4h
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period volume average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Calculate 4h ATR(14) for stop loss
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(lookback, vol_period, 14) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Oversold reversal with volume and above weekly EMA50
            if williams_r[i] < -80 and vol_filter and price > ema_50_1w_aligned[i]:
                signals[i] = size
                position = 1
            # Short: Overbought reversal with volume and below weekly EMA50
            elif williams_r[i] > -20 and vol_filter and price < ema_50_1w_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Overbought or trailing stop
            if williams_r[i] > -20 or price < ema_50_1w_aligned[i] - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Oversold or trailing stop
            if williams_r[i] < -80 or price > ema_50_1w_aligned[i] + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_WilliamsR_14_1wEMA50_Volume"
timeframe = "4h"
leverage = 1.0