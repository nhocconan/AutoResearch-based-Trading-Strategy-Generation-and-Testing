#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_trix_volume_regime_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Calculate 1d TRIX (15-period)
    close_1d = df_1d['close'].values
    # First EMA
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Second EMA
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Third EMA
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # TRIX = (ema3 - previous ema3) / previous ema3 * 100
    trix_raw = np.zeros_like(ema3)
    trix_raw[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    trix_raw[0] = 0.0
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # TRIX histogram (main signal)
    trix_hist = trix_raw - trix_signal
    
    # Align TRIX histogram to 4h timeframe
    trix_hist_aligned = align_htf_to_ltf(prices, df_1d, trix_hist)
    
    # 4h ATR for volatility filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(trix_hist_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        trix_val = trix_hist_aligned[i]
        vol_ma = vol_ma_20[i]
        volume_current = volume[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.3 * vol_ma
        
        # TRIX-based signals with volume confirmation
        long_signal = trix_val > 0.05 and volume_confirmed
        short_signal = trix_val < -0.05 and volume_confirmed
        
        # Exit when TRIX crosses zero (mean reversion)
        exit_long = position == 1 and trix_val < 0
        exit_short = position == -1 and trix_val > 0
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: TRIX momentum with volume confirmation on 4h timeframe using 1d TRIX.
# Uses 1d TRIX (15,9,9) to identify momentum shifts on higher timeframe.
# Enters long when 1d TRIX histogram > 0.05 with volume confirmation (>1.3x avg volume).
# Enters short when 1d TRIX histogram < -0.05 with volume confirmation.
# Exits when TRIX histogram crosses zero (mean reversion).
# The volume filter ensures trades occur with conviction, reducing false signals.
# Works in both bull and bear markets by capturing momentum shifts in either direction.
# Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.
# Uses discrete position sizing (0.25) to reduce churn and transaction costs.