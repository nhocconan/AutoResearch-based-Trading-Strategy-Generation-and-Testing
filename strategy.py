#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for ATR and Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR(14)
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]
    tr2[0] = np.abs(high_1w[0] - close_1w[0])
    tr3[0] = np.abs(low_1w[0] - close_1w[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate weekly Bollinger Bands (20, 2.0)
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb_1w = sma_20_1w + 2.0 * std_20_1w
    lower_bb_1w = sma_20_1w - 2.0 * std_20_1w
    
    # Align weekly indicators to 4h timeframe
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    upper_bb_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_bb_1w)
    lower_bb_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_bb_1w)
    
    # Calculate 4h ATR for stop loss
    tr_4h_1 = high - low
    tr_4h_2 = np.abs(high - np.roll(close, 1))
    tr_4h_3 = np.abs(low - np.roll(close, 1))
    tr_4h_1[0] = high[0] - low[0]
    tr_4h_2[0] = np.abs(high[0] - close[0])
    tr_4h_3[0] = np.abs(low[0] - close[0])
    tr_4h = np.maximum(tr_4h_1, np.maximum(tr_4h_2, tr_4h_3))
    atr_4h = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # need weekly BB, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_1w_aligned[i]) or np.isnan(upper_bb_1w_aligned[i]) or 
            np.isnan(lower_bb_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price below weekly lower Bollinger Band + 0.5*ATR, with volume
            if (close[i] < lower_bb_1w_aligned[i] + 0.5 * atr_1w_aligned[i] and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short entry: price above weekly upper Bollinger Band - 0.5*ATR, with volume
            elif (close[i] > upper_bb_1w_aligned[i] - 0.5 * atr_1w_aligned[i] and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses above weekly upper Bollinger Band or ATR-based stop
            if close[i] > upper_bb_1w_aligned[i] - 0.5 * atr_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses below weekly lower Bollinger Band or ATR-based stop
            if close[i] < lower_bb_1w_aligned[i] + 0.5 * atr_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WeeklyBBand_Touch_VolumeFilter"
timeframe = "4h"
leverage = 1.0