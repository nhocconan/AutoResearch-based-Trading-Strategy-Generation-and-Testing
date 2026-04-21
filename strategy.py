#!/usr/bin/env python3
"""
6h_IBS_Regime_Filter_VolumeSpike
Hypothesis: 6h IBS (Internal Bar Strength) combined with 1d trend regime (EMA50) and volume confirmation.
IBS = (close - low) / (high - low) measures where close falls within the bar's range.
Long when IBS < 0.3 (oversold) in bullish regime (price > 1d EMA50); short when IBS > 0.7 (overbought) in bearish regime (price < 1d EMA50).
Volume confirmation (>1.5x average) filters low-quality signals.
Designed to work in both bull and bear markets via regime alignment and mean-reversion logic.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend regime ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h IBS calculation ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Avoid division by zero
    hl_range = high - low
    ibs = np.where(hl_range != 0, (close - low) / hl_range, 0.5)
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ibs_val = ibs[i]
        volume_now = volume[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: oversold IBS in bullish regime
            long_condition = (ibs_val < 0.3) and (price > ema_trend) and volume_confirmed
            # Short: overbought IBS in bearish regime
            short_condition = (ibs_val > 0.7) and (price < ema_trend) and volume_confirmed
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions for long
            if ibs_val > 0.7:  # IBS reversal signal
                signals[i] = 0.0
                position = 0
            elif price < ema_trend:  # regime change
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions for short
            if ibs_val < 0.3:  # IBS reversal signal
                signals[i] = 0.0
                position = 0
            elif price > ema_trend:  # regime change
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_IBS_Regime_Filter_VolumeSpike"
timeframe = "6h"
leverage = 1.0