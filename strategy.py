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
    
    # Load weekly data for EMA50 trend filter (weekly EMA50 = ~50 weeks trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly EMA50 for trend
    ema_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * 2 + ema_50_1w[i-1] * 48) / 50
    
    # Calculate weekly ATR for volatility filter
    atr_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 14:
        tr = np.zeros(len(close_1w))
        for i in range(1, len(close_1w)):
            tr[i] = max(
                high_1w[i] - low_1w[i],
                abs(high_1w[i] - close_1w[i-1]),
                abs(low_1w[i] - close_1w[i-1])
            )
        # Wilder's smoothing for ATR
        atr_1w[13] = np.mean(tr[1:14])
        for i in range(14, len(close_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
    
    # Calculate weekly Bollinger Bands (20, 2.0)
    bb_upper_1w = np.full_like(close_1w, np.nan)
    bb_lower_1w = np.full_like(close_1w, np.nan)
    bb_middle_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 20:
        for i in range(19, len(close_1w)):
            bb_middle_1w[i] = np.mean(close_1w[i-19:i+1])
            bb_std = np.std(close_1w[i-19:i+1])
            bb_upper_1w[i] = bb_middle_1w[i] + 2.0 * bb_std
            bb_lower_1w[i] = bb_middle_1w[i] - 2.0 * bb_std
    
    # Align weekly indicators to 12h timeframe
    ema_50_1w_12h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    atr_1w_12h = align_htf_to_ltf(prices, df_1w, atr_1w)
    bb_upper_1w_12h = align_htf_to_ltf(prices, df_1w, bb_upper_1w)
    bb_lower_1w_12h = align_htf_to_ltf(prices, df_1w, bb_lower_1w)
    bb_middle_1w_12h = align_htf_to_ltf(prices, df_1w, bb_middle_1w)
    
    # Volume spike detection on 12h bars
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1w_12h[i]) or 
            np.isnan(atr_1w_12h[i]) or
            np.isnan(bb_upper_1w_12h[i]) or
            np.isnan(bb_lower_1w_12h[i]) or
            np.isnan(bb_middle_1w_12h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 12h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Price position relative to weekly Bollinger Bands
        bb_position = (close[i] - bb_lower_1w_12h[i]) / (bb_upper_1w_12h[i] - bb_lower_1w_12h[i]) if (bb_upper_1w_12h[i] - bb_lower_1w_12h[i]) > 0 else 0.5
        
        if position == 0:
            # Long: Price near lower BB (oversold) with volume spike and above weekly EMA50
            if (bb_position < 0.2 and  # Near lower Bollinger Band
                volume_ratio > 2.0 and  # Volume spike
                close[i] > ema_50_1w_12h[i]):  # Above weekly trend
                position = 1
                signals[i] = position_size
            # Short: Price near upper BB (overbought) with volume spike and below weekly EMA50
            elif (bb_position > 0.8 and   # Near upper Bollinger Band
                  volume_ratio > 2.0 and  # Volume spike
                  close[i] < ema_50_1w_12h[i]):  # Below weekly trend
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price crosses back to middle BB or below weekly EMA50
            if (bb_position > 0.5 or  # Back to middle or upper BB
                close[i] < ema_50_1w_12h[i]):  # Below weekly trend
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price crosses back to middle BB or above weekly EMA50
            if (bb_position < 0.5 or   # Back to middle or lower BB
                close[i] > ema_50_1w_12h[i]):  # Above weekly trend
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_BB_EMA50_Volume"
timeframe = "12h"
leverage = 1.0