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
    
    # Get weekly data for calculations (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly RSI(14) for trend filter
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Calculate RSI with proper smoothing
    avg_gain = np.full(len(close_1w), np.nan)
    avg_loss = np.full(len(close_1w), np.nan)
    rsi = np.full(len(close_1w), np.nan)
    
    # Wilder's smoothing
    for i in range(14, len(close_1w)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    # Calculate weekly ATR(14) for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # First TR is undefined
    
    atr = np.full(len(close_1w), np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate weekly Bollinger Bands (20, 2.0)
    sma_20 = np.full(len(close_1w), np.nan)
    std_20 = np.full(len(close_1w), np.nan)
    upper_band = np.full(len(close_1w), np.nan)
    lower_band = np.full(len(close_1w), np.nan)
    
    for i in range(19, len(close_1w)):
        sma_20[i] = np.mean(close_1w[i-19:i+1])
        std_20[i] = np.std(close_1w[i-19:i+1])
        upper_band[i] = sma_20[i] + 2 * std_20[i]
        lower_band[i] = sma_20[i] - 2 * std_20[i]
    
    # Align weekly indicators to daily timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    atr_aligned = align_htf_to_ltf(prices, df_1w, atr)
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    
    # Calculate 10-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 10
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(20, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        # Weekly Bollinger Band squeeze detection (BB width < 1 ATR)
        bb_width = upper_band_aligned[i] - lower_band_aligned[i]
        squeeze_filter = bb_width < atr_aligned[i]
        
        if position == 0:
            # Long: Price breaks above upper BB with volume, RSI > 50, and BB squeeze
            if price > upper_band_aligned[i] and vol_filter and rsi_aligned[i] > 50 and squeeze_filter:
                signals[i] = size
                position = 1
            # Short: Price breaks below lower BB with volume, RSI < 50, and BB squeeze
            elif price < lower_band_aligned[i] and vol_filter and rsi_aligned[i] < 50 and squeeze_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below lower BB or RSI < 40
            if price < lower_band_aligned[i] or rsi_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above upper BB or RSI > 60
            if price > upper_band_aligned[i] or rsi_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyBB_Squeeze_RSI_Volume"
timeframe = "1d"
leverage = 1.0