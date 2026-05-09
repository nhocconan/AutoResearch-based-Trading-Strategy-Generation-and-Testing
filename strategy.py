#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_Volume_Spike_Regime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    """
    TRIX momentum with volume spike and chop regime filter.
    - TRIX(12) crosses above signal line (EMA9 of TRIX) + volume spike + chop < 61.8 (trending regime) -> long
    - TRIX(12) crosses below signal line + volume spike + chop < 61.8 -> short
    - Exit when TRIX crosses back through zero line
    - Uses volume spike: current volume > 2.0 x 20-period average
    - Chop regime filter: CHOP(14) < 61.8 indicates trending market
    - Designed for fewer trades (target: 20-40/year) with strong edge in trending markets
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate TRIX: EMA(EMA(EMA(close, 12), 12), 12)
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.pct_change() * 100  # percentage change
    
    # Signal line: EMA of TRIX, period 9
    trix_signal = trix.ewm(span=9, adjust=False, min_periods=9).mean()
    
    # Chop regime: CHOP(14) = 100 * log15(ATR(14) / (HHH(14) - LLL(14))) / log15(14)
    # Simplified: CHOP = 100 * log(ATR(14) / (max_high - min_low)) / log(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    
    max_high14 = pd.Series(high).rolling(window=14, min_periods=14).max()
    min_low14 = pd.Series(low).rolling(window=14, min_periods=14).min()
    
    # Avoid division by zero
    range14 = max_high14 - min_low14
    range14 = np.where(range14 == 0, 1e-10, range14)
    
    chop = 100 * np.log10(atr14 / range14) / np.log10(14)
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma20.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix.iloc[i]) if hasattr(trix, 'iloc') else np.isnan(trix[i]) or
            np.isnan(trix_signal.iloc[i]) if hasattr(trix_signal, 'iloc') else np.isnan(trix_signal[i]) or
            np.isnan(chop[i]) or np.isnan(vol_ma20.iloc[i]) if hasattr(vol_ma20, 'iloc') else np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Extract values safely
        trix_val = trix.iloc[i] if hasattr(trix, 'iloc') else trix[i]
        trix_signal_val = trix_signal.iloc[i] if hasattr(trix_signal, 'iloc') else trix_signal[i]
        chop_val = chop[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: TRIX crosses above signal line + volume spike + trending regime (chop < 61.8)
            if trix_val > trix_signal_val and vol_spike_val and chop_val < 61.8:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below signal line + volume spike + trending regime
            elif trix_val < trix_signal_val and vol_spike_val and chop_val < 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero (momentum fade)
            if trix_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero
            if trix_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals