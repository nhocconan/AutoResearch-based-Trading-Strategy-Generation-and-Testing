#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band breakout with 1d ATR filter and volume spike confirmation.
- Primary timeframe: 6h for entries/exits.
- HTF: 1d ATR (14-period) to filter breakouts (only trade when ATR > 20-period ATR MA).
- Volume: Current 6h volume > 1.5 * 20-period volume MA to confirm momentum.
- Entry: Long when price breaks above upper BB(20,2) AND 1d ATR filter bullish AND volume spike.
         Short when price breaks below lower BB(20,2) AND 1d ATR filter bearish AND volume spike.
- Exit: Price returns to middle BB (20-period SMA) or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
BB breakouts capture volatility expansion; ATR filter ensures we trade only in sufficient volatility regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands on 6h
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2.0 * std_20)
    lower_bb = sma_20 - (2.0 * std_20)
    middle_bb = sma_20
    
    # Get 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR (14-period)
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range
    tr1 = df_1d_high - df_1d_low
    tr2 = np.abs(df_1d_high - np.roll(df_1d_close, 1))
    tr3 = np.abs(df_1d_low - np.roll(df_1d_close, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period ATR MA for filter
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    
    # ATR filter: 1 if bullish (current ATR > MA), -1 if bearish (current ATR < MA), 0 otherwise
    atr_filter = np.where(atr_14 > atr_ma_20, 1, np.where(atr_14 < atr_ma_20, -1, 0))
    
    # Calculate 20-period volume MA on 1d
    df_1d_volume = df_1d['volume'].values
    vol_ma_20 = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Volume confirmation: current 6h volume > 1.5 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.5 * vol_ma_20_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 30)  # Need enough bars for BB and 1d indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or 
            np.isnan(atr_filter_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        atr_val = atr_filter_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: price breaks above upper BB AND 1d ATR filter bullish
                if curr_close > upper_bb[i] and atr_val == 1:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below lower BB AND 1d ATR filter bearish
                elif curr_close < lower_bb[i] and atr_val == -1:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price returns to middle BB OR loss of volume confirmation
            if curr_close <= middle_bb[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to middle BB OR loss of volume confirmation
            if curr_close >= middle_bb[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BollingerBreakout_1dATRFilter_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0