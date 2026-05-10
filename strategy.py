#!/usr/bin/env python3
# 4H_Modified_Chaikin_Oscillator_With_ADX_Filter
# Hypothesis: Chaikin Oscillator (3,10) measures money flow momentum. When combined with ADX(14) to filter ranging markets (ADX<20) and using volume confirmation (volume > 1.5x average), it captures momentum shifts in both bull and bear markets. The strategy avoids false signals in low-volatility periods and focuses on high-conviction moves. Designed for low frequency (~20-40 trades/year) to minimize fee drag.

name = "4H_Modified_Chaikin_Oscillator_With_ADX_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Money Flow Multiplier and Money Flow Volume
    mfm = np.where((high - low) != 0, ((close - low) - (high - close)) / (high - low), 0)
    mfv = mfm * volume
    
    # Chaikin Oscillator: (3-period EMA of MFV) - (10-period EMA of MFV)
    mfv_series = pd.Series(mfv)
    ema3_mfv = mfv_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10_mfv = mfv_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    chaikin_osc = ema3_mfv - ema10_mfv
    
    # ADX calculation (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # first value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = np.where(tr_sum != 0, 100 * dm_plus_sum / tr_sum, 0)
    di_minus = np.where(tr_sum != 0, 100 * dm_minus_sum / tr_sum, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(chaikin_osc[i]) or np.isnan(adx[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        trending = adx[i] > 20  # Only trade in trending markets
        
        if position == 0:
            # Enter long: positive Chaikin Oscillator with volume and trend
            if (chaikin_osc[i] > 0 and volume_confirm and trending):
                signals[i] = 0.25
                position = 1
            # Enter short: negative Chaikin Oscillator with volume and trend
            elif (chaikin_osc[i] < 0 and volume_confirm and trending):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when Chaikin Oscillator turns negative or trend weakens
            if (chaikin_osc[i] < 0 or adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when Chaikin Oscillator turns positive or trend weakens
            if (chaikin_osc[i] > 0 or adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals