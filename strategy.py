#!/usr/bin/env python3
# 4h_1d_trix_volume_chop
# Hypothesis: 4-hour TRIX (12-period) with volume confirmation and Choppiness index regime filter.
# TRIX filters noise and captures momentum shifts. Volume confirms breakout strength.
# Chop filter avoids trend-following in ranging markets (Chop > 61.8) and avoids mean-reversion in strong trends (Chop < 38.2).
# Works in bull/bear by adapting to regime: trend-following when Chop < 38.2, mean-reversion when Chop > 61.8.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.

name = "4h_1d_trix_volume_chop"
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
    
    # Get daily data for TRIX and Choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # TRIX: triple EMA of log returns
    # Step 1: EMA1 of close
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean()
    # Step 2: EMA2 of EMA1
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    # Step 3: EMA3 of EMA2
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    # TRIX = (EMA3 - prev EMA3) / prev EMA3 * 100
    trix_raw = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    trix = trix_raw.fillna(0).values
    
    # Choppiness Index (14-period)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    chop = chop.fillna(50).values  # neutral when undefined
    
    # Align TRIX and Chop to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime-based logic
        if chop_aligned[i] < 38.2:  # Trending regime
            # Trend following: TRIX momentum
            if trix_aligned[i] > 0.1 and vol_confirm[i] and position != 1:
                position = 1
                signals[i] = 0.25
            elif trix_aligned[i] < -0.1 and vol_confirm[i] and position != -1:
                position = -1
                signals[i] = -0.25
            # Exit on TRIX reversal
            elif position == 1 and trix_aligned[i] < 0:
                position = 0
                signals[i] = 0.0
            elif position == -1 and trix_aligned[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                # Hold position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
                    
        elif chop_aligned[i] > 61.8:  # Ranging regime
            # Mean reversion: TRIX extreme
            if trix_aligned[i] < -0.2 and vol_confirm[i] and position != 1:
                position = 1
                signals[i] = 0.25
            elif trix_aligned[i] > 0.2 and vol_confirm[i] and position != -1:
                position = -1
                signals[i] = -0.25
            # Exit on TRIX return to zero
            elif position == 1 and trix_aligned[i] > -0.05:
                position = 0
                signals[i] = 0.0
            elif position == -1 and trix_aligned[i] < 0.05:
                position = 0
                signals[i] = 0.0
            else:
                # Hold position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:  # Neutral regime (38.2 <= Chop <= 61.8)
            # No position in uncertain regime
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals