#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_TRIX_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    """
    6h Elder Ray (Bull/Bear Power) with TRIX filter and volume confirmation.
    - Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    - TRIX(12) filter: TRIX > 0 = bullish bias, TRIX < 0 = bearish bias
    - Long: Bull Power > 0, TRIX > 0, volume > 1.5x avg
    - Short: Bear Power < 0, TRIX < 0, volume > 1.5x avg
    - Exit: Opposite Elder Ray signal or TRIX crosses zero
    - Target: 20-40 trades/year on 6h timeframe
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA(13) for Elder Ray
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate TRIX(12, 9) for trend filter
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - then % change
    ema1 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix_raw = ema3.pct_change(periods=1) * 100  # Percentage change
    trix = trix_raw.fillna(0).values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # ensure sufficient warmup for EMA13 and TRIX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(trix[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: Bull Power positive, TRIX positive, volume confirmation
            if bull_power[i] > 0 and trix[i] > 0 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative, TRIX negative, volume confirmation
            elif bear_power[i] < 0 and trix[i] < 0 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bear Power negative or TRIX crosses below zero
            if bear_power[i] < 0 or trix[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bull Power positive or TRIX crosses above zero
            if bull_power[i] > 0 or trix[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals