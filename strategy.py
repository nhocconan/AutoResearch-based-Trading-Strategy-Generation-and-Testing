#!/usr/bin/env python3
# 12h_trix_volume_regime_v1
# Hypothesis: On 12h timeframe, capture momentum reversals using TRIX crossing zero with volume confirmation and Choppiness regime filter.
# TRIX > 0 indicates bullish momentum, TRIX < 0 bearish. Volume spike confirms breakout strength.
# Chop > 61.8 identifies ranging markets where TRIX crossovers are more reliable.
# Works in bull/bear by following momentum direction with volume confirmation.
# Target: 20-40 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_trix_volume_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter (1d HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # TRIX (12-period)
    # EMA1
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA2
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA3
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # TRIX = 100 * (EMA3 - previous EMA3) / previous EMA3
    trix = np.zeros(n)
    trix[1:] = 100 * (ema3[1:] - ema3[:-1]) / ema3[:-1]
    trix[0] = 0
    
    # Volume spike detector (volume > 1.5 * 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    # Choppiness Index (14-period)
    high = prices['high'].values
    low = prices['low'].values
    
    tr1 = high - low
    tr2 = np.abs(np.concatenate([[high[0]], high[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(np.concatenate([[low[0]], low[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((atr * 14) / (max_high - min_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trix[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_spike[i]) or np.isnan(chop[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: TRIX crosses below zero OR volume dries up OR chop < 38.2 (strong trend)
            if (trix[i] < 0 and trix[i-1] >= 0) or (not vol_spike[i]) or (chop[i] < 38.2):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX crosses above zero OR volume dries up OR chop < 38.2
            if (trix[i] > 0 and trix[i-1] <= 0) or (not vol_spike[i]) or (chop[i] < 38.2):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Require Chop > 61.8 (ranging market) and volume spike
            if chop[i] > 61.8 and vol_spike[i]:
                # Long entry: TRIX crosses above zero AND above daily EMA50 (bullish alignment)
                if (trix[i] > 0 and trix[i-1] <= 0) and (close[i] > ema_50_1d_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short entry: TRIX crosses below zero AND below daily EMA50 (bearish alignment)
                elif (trix[i] < 0 and trix[i-1] >= 0) and (close[i] < ema_50_1d_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals