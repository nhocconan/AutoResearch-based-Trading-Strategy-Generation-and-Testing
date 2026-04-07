#!/usr/bin/env python3
"""
4h_trix_volume_regime_v1
Hypothesis: TRIX (triple exponential average) combined with volume confirmation and 
choppiness regime filter works on 4h timeframe. TRIX > 0 indicates bullish momentum, 
TRIX < 0 indicates bearish momentum. Volume confirms breakouts. Choppiness filter 
avoids sideways markets where TRIX whipsaws. Works in both bull and bear by 
adjusting to regime - in choppy markets we avoid trades, in trending we follow TRIX.
Targets 20-50 trades/year (80-200 over 4 years). Uses TRIX(12) with signal line 
(9-period EMA of TRIX) for entry/exit.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_trix_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate TRIX on close prices
    # TRIX = EMA(EMA(EMA(close, period), period), period) 
    # Then % change: (today's TRIX - yesterday's TRIX) / yesterday's TRIX * 100
    ema1 = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema2 = ema1.ewm(span=12, adjust=False).mean()
    ema3 = ema2.ewm(span=12, adjust=False).mean()
    trix_raw = ema3.pct_change() * 100  # Percentage change
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = trix_raw.ewm(span=9, adjust=False).mean()
    
    # Histogram for crossover signals
    trix_hist = trix_raw - trix_signal
    
    # Calculate choppiness index for regime filter
    # CHOP = 100 * log10(sum(ATR over n) / (log10(highest high - lowest low) * n))
    # Simplified: high-low range over period vs true range
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean()
    hh = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max()
    ll = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min()
    
    # Avoid division by zero
    range_hl = hh - ll
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    chop = 100 * np.log10(atr * atr_period / range_hl) / np.log10(atr_period)
    
    # Volume confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(trix_raw.iloc[i]) or np.isnan(trix_signal.iloc[i]) or 
            np.isnan(trix_hist.iloc[i]) or np.isnan(chop.iloc[i]) or 
            np.isnan(vol_sma.iloc[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma.iloc[i]
        
        # Choppiness regime: only trade when trending (CHOP < 50)
        # CHOP > 61.8 = ranging, CHOP < 38.2 = strongly trending
        # We use middle ground: avoid extreme chop (CHOP > 50)
        if chop.iloc[i] > 50:
            # In choppy markets, stay flat
            position = 0
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: TRIX crosses below signal line OR TRIX turns negative
            if trix_hist.iloc[i] < 0 or trix_raw.iloc[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: TRIX crosses above signal line OR TRIX turns positive
            if trix_hist.iloc[i] > 0 or trix_raw.iloc[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: TRIX crosses above signal line with volume confirmation
            if (trix_hist.iloc[i] > 0 and trix_hist.iloc[i-1] <= 0 and 
                vol_confirm and trix_raw.iloc[i] > 0):
                position = 1
                signals[i] = 0.25
            # Short: TRIX crosses below signal line with volume confirmation
            elif (trix_hist.iloc[i] < 0 and trix_hist.iloc[i-1] >= 0 and 
                  vol_confirm and trix_raw.iloc[i] < 0):
                position = -1
                signals[i] = -0.25
    
    return signals