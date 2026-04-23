#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band squeeze breakout with 1d ATR regime filter and volume confirmation.
Long when price breaks above upper BB(20,2) AND BB width < 20th percentile (squeeze) AND 1d ATR > 15th percentile (volatile regime) AND volume > 2x 20-period average.
Short when price breaks below lower BB(20,2) AND same conditions.
Exit when price re-enters the Bollinger Bands or BB width expands above 50th percentile.
Uses 1d HTF for ATR regime to ensure breakouts occur in sufficient volatility. Target: 50-150 total trades over 4 years (12-37/year).
Bollinger Bands from 20-period SMA and 2 std dev. BB width = (upper - lower) / middle.
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
    
    # Calculate Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + bb_std * std
    lower_band = sma - bb_std * std
    bb_width = (upper_band - lower_band) / sma  # normalized width
    
    # Calculate 1d ATR(14) for regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate percentiles for BB width and ATR regime (using expanding window to avoid look-ahead)
    bb_width_percentile = np.full(n, np.nan)
    atr_percentile = np.full(n, np.nan)
    
    for i in range(bb_period, n):
        # BB width percentile: rank of current value among historical values
        historical_width = bb_width[bb_period:i+1]
        if len(historical_width) > 0:
            current_width = bb_width[i]
            bb_width_percentile[i] = (np.sum(historical_width <= current_width) / len(historical_width)) * 100
        
        # ATR percentile: rank of current ATR among historical values
        if not np.isnan(atr_14_aligned[i]):
            historical_atr = atr_14_aligned[bb_period:i+1]
            historical_atr = historical_atr[~np.isnan(historical_atr)]
            if len(historical_atr) > 0:
                current_atr = atr_14_aligned[i]
                atr_percentile[i] = (np.sum(historical_atr <= current_atr) / len(historical_atr)) * 100
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(bb_period, 20)  # BB (20), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sma[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(bb_width_percentile[i]) or np.isnan(atr_percentile[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        bb_width_pct = bb_width_percentile[i]
        atr_pct = atr_percentile[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Break above upper BB AND BB squeeze (width < 20th percentile) AND volatile regime (ATR > 15th percentile) AND volume spike
            if (price > upper_band[i] and 
                bb_width_pct < 20.0 and 
                atr_pct > 15.0 and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Break below lower BB AND same conditions
            elif (price < lower_band[i] and 
                  bb_width_pct < 20.0 and 
                  atr_pct > 15.0 and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price re-enters BB OR BB width expands above 50th percentile (squeeze end)
                if price < upper_band[i] and price > lower_band[i]:
                    exit_signal = True
                elif bb_width_pct > 50.0:
                    exit_signal = True
            elif position == -1:
                # Short exit: price re-enters BB OR BB width expands above 50th percentile
                if price < upper_band[i] and price > lower_band[i]:
                    exit_signal = True
                elif bb_width_pct > 50.0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_BollingerSqueezebreakout_ATRregime_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0