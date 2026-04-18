#!/usr/bin/env python3
"""
6h_1d_Volume_Weighted_RSI_Momentum
Hypothesis: Use RSI(14) from 1h data as momentum filter, combined with 1d VWAP deviation and volume spike on 6h. 
In bull markets: RSI > 55 + price > 1d VWAP + volume > 2x average → long.
In bear markets: RSI < 45 + price < 1d VWAP + volume > 2x average → short.
Volume spike filters low-conviction moves; VWAP acts as dynamic support/resistance.
Targets 15-25 trades/year by requiring triple confirmation.
"""

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
    
    # Get 1h data for RSI (lower TF for timing)
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    
    # Calculate RSI(14) on 1h
    def rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        return 100 - (100 / (1 + rs))
    
    rsi_1h = rsi(close_1h, 14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Get 1d data for VWAP
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate typical price and VWAP
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    cum_vol_tp = np.cumsum(typical_price_1d * volume_1d)
    cum_vol = np.cumsum(volume_1d)
    vwap_1d = np.divide(cum_vol_tp, cum_vol, out=np.full_like(cum_vol_tp, np.nan), where=cum_vol!=0)
    
    # Align VWAP to 6h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Volume confirmation: current volume > 2 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # need volume MA and RSI warmed up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_1h_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: RSI > 55, price above VWAP, volume spike
            if (rsi_1h_aligned[i] > 55 and 
                close[i] > vwap_1d_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: RSI < 45, price below VWAP, volume spike
            elif (rsi_1h_aligned[i] < 45 and 
                  close[i] < vwap_1d_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: RSI < 50 (momentum fade) or price below VWAP (support break)
            if (rsi_1h_aligned[i] < 50 or 
                close[i] < vwap_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI > 50 (momentum fade) or price above VWAP (resistance break)
            if (rsi_1h_aligned[i] > 50 or 
                close[i] > vwap_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_Volume_Weighted_RSI_Momentum"
timeframe = "6h"
leverage = 1.0