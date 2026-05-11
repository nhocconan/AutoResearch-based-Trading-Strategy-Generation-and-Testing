#!/usr/bin/env python3
name = "4h_Bollinger_Band_Width_Regime_Keltner_Breakout"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Band Width for regime detection (20-period)
    ma20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper_bb = ma20 + 2 * std20
    lower_bb = ma20 - 2 * std20
    bb_width = (upper_bb - lower_bb) / ma20
    bb_width = np.nan_to_num(bb_width, nan=0.0)
    
    # Bollinger Band Width percentile (50-period) for regime
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).rank(pct=True) * 100
    bb_width_percentile = bb_width_percentile.fillna(50).values
    
    # Keltner Channel (20-period, ATR multiplier 1.5)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean()
    ma20_atr = pd.Series(close).rolling(window=20, min_periods=20).mean()
    upper_keltner = ma20_atr + 1.5 * atr
    lower_keltner = ma20_atr - 1.5 * atr
    
    # Volume surge filter (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(bb_width_percentile[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Regime filter: Bollinger Band Width percentile
        # Range: BB width percentile > 50 (high volatility)
        # Trend: BB width percentile <= 50 (low volatility)
        is_range = bb_width_percentile[i] > 50
        is_trend = bb_width_percentile[i] <= 50
        
        if position == 0:
            # Long entry: price breaks above upper Keltner in range regime with volume
            if (is_range and 
                close[i] > upper_keltner[i] and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Keltner in range regime with volume
            elif (is_range and 
                  close[i] < lower_keltner[i] and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to middle Keltner line (mean reversion)
            if position == 1:
                if close[i] <= ma20_atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if close[i] >= ma20_atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals