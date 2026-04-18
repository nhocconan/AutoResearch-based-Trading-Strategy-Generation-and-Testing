#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Bollinger Bands and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on daily data
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate Bollinger Bands on daily close (20-period, 2 std)
    sma_20 = np.full(len(close_1d), np.nan)
    std_20 = np.full(len(close_1d), np.nan)
    for i in range(20, len(close_1d)):
        sma_20[i] = np.mean(close_1d[i-20:i])
        std_20[i] = np.std(close_1d[i-20:i])
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    # Align daily indicators to 1d timeframe (no additional delay needed for BB)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Calculate 1d ATR for stop loss and position sizing
    tr_1d_1 = high - low
    tr_1d_2 = np.abs(high - np.roll(close, 1))
    tr_1d_3 = np.abs(low - np.roll(close, 1))
    tr_1d_1[0] = high[0] - low[0]
    tr_1d_2[0] = np.abs(high[0] - close[0])
    tr_1d_3[0] = np.abs(low[0] - close[0])
    tr_1d = np.maximum(tr_1d_1, np.maximum(tr_1d_2, tr_1d_3))
    atr_1d_current = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # need Bollinger Bands and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr_1d_current[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price touches or breaks below lower Bollinger Band with volume
            if close[i] <= lower_band_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price touches or breaks above upper Bollinger Band with volume
            elif close[i] >= upper_band_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns to middle Bollinger Band (SMA20) or ATR stop
            if close[i] >= sma_20_aligned[i] if 'sma_20_aligned' in locals() else False:
                signals[i] = 0.0
                position = 0
            elif close[i] < open_price[i] - 2.0 * atr_1d_current[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle Bollinger Band (SMA20) or ATR stop
            if close[i] <= sma_20_aligned[i] if 'sma_20_aligned' in locals() else False:
                signals[i] = 0.0
                position = 0
            elif close[i] > open_price[i] + 2.0 * atr_1d_current[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_BollingerBandReversal_VolumeFilter"
timeframe = "1d"
leverage = 1.0