#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 12h/1d confluence - Long when price > 12h EMA(20) AND price > 1d VWAP with volume > 1.5x avg volume.
# Short when price < 12h EMA(20) AND price < 1d VWAP with volume > 1.5x avg volume.
# Uses VWAP for institutional reference and EMA for trend alignment. Volume confirms institutional participation.
# Target: 25-40 trades/year (100-160 total over 4 years) for balanced frequency.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate EMA(20) on 12h
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 20:
        ema_12h[19] = np.mean(close_12h[:20])
        for i in range(20, len(close_12h)):
            ema_12h[i] = (close_12h[i] * 0.0952) + (ema_12h[i-1] * 0.9048)
    
    # 1d data for VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate VWAP (typical price * volume) cumulative
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_1d = np.full(len(close_1d), np.nan)
    cum_tpv = 0.0
    cum_vol = 0.0
    for i in range(len(close_1d)):
        cum_tpv += typical_price_1d[i] * volume_1d[i]
        cum_vol += volume_1d[i]
        if cum_vol > 0:
            vwap_1d[i] = cum_tpv / cum_vol
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 12h EMA and 1d VWAP to 4h
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema = ema_12h_aligned[i]
        vwap = vwap_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price > EMA AND price > VWAP + volume confirmation
            if (price > ema and price > vwap and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price < EMA AND price < VWAP + volume confirmation
            elif (price < ema and price < vwap and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price < EMA OR price < VWAP
            if price < ema or price < vwap:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price > EMA OR price > VWAP
            if price > ema or price > vwap:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_1d_EMA_VWAP_Volume"
timeframe = "4h"
leverage = 1.0