#!/usr/bin/env python3
# 6h_adx_di_volume_v3
# Hypothesis: 6h strategy using 1d ADX and DI crossover for trend strength and direction, combined with 6h volume confirmation.
# ADX > 25 indicates strong trend; DI+ > DI- for bullish, DI- > DI+ for bearish.
# Volume > 1.5x 20-period average filters weak moves.
# Discrete sizing (±0.25) to minimize fee churn. Target: 50-150 total trades over 4 years (12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_di_volume_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for ADX/DI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX/DI calculation (Wilder's smoothing)
    period = 14
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = -pd.Series(low_1d).diff()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)
    
    # Smoothed TR and DM
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    dm_plus_smooth = dm_plus.ewm(alpha=1/period, adjust=False).mean()
    dm_minus_smooth = dm_minus.ewm(alpha=1/period, adjust=False).mean()
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    # Align to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    di_plus_aligned = align_htf_to_ltf(prices, df_1d, di_plus.values)
    di_minus_aligned = align_htf_to_ltf(prices, df_1d, di_minus.values)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(di_plus_aligned[i]) or 
            np.isnan(di_minus_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ADX < 20 (trend weakening) or DI- > DI+ (trend reversal)
            if adx_aligned[i] < 20.0 or di_minus_aligned[i] > di_plus_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ADX < 20 (trend weakening) or DI+ > DI- (trend reversal)
            if adx_aligned[i] < 20.0 or di_plus_aligned[i] > di_minus_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation and strong trend
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            strong_trend = adx_aligned[i] > 25.0
            
            if volume_confirmed and strong_trend:
                # Bullish: DI+ > DI-
                if di_plus_aligned[i] > di_minus_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Bearish: DI- > DI+
                elif di_minus_aligned[i] > di_plus_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals