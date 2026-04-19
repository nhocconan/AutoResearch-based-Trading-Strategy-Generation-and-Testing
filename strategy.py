#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX momentum with volume confirmation and Choppiness regime filter.
# TRIX (12-period) filters noise and captures momentum shifts; long when TRIX > 0 and rising, short when TRIX < 0 and falling.
# Volume spike (>1.5x 20-period average) confirms breakout strength.
# Choppiness Index (14-period) > 61.8 indicates ranging market (avoid trend signals); < 38.2 indicates trending (favor TRIX signals).
# Designed for 4h timeframe to capture medium-term momentum with low frequency (~20-40 trades/year).
# Entry: Long when TRIX > 0, TRIX rising, volume spike, and CHOP < 38.2; Short when TRIX < 0, TRIX falling, volume spike, and CHOP < 38.2.
# Exit: Opposite TRIX signal or CHOP > 61.8 (range mode).
# Uses strict conditions to limit trades and avoid overtrading.

name = "4h_TRIX_Volume_Chop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # TRIX: triple EMA of ROC, period=12
    close_series = pd.Series(close)
    roc = close_series.pct_change(1)
    ema1 = roc.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100  # percent change
    trix = trix.values
    
    # TRIX rising/falling: compare to prior value
    trix_rising = trix > np.roll(trix, 1)
    trix_falling = trix < np.roll(trix, 1)
    # Handle first element
    trix_rising[0] = False
    trix_falling[0] = False
    
    # Volume spike: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    # Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR, 14) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = max_high - min_low
    # Avoid division by zero
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = 100 * np.log10(atr_sum / range_hl) / np.log10(14)
    chop = np.where(np.isnan(chop), 50.0, chop)  # neutral if undefined
    
    # Regime filters: trending when CHOP < 38.2, ranging when CHOP > 61.8
    trending = chop < 38.2
    ranging = chop > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(trix_rising[i]) or np.isnan(trix_falling[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX positive and rising, volume spike, trending market
            if (trix[i] > 0 and trix_rising[i] and volume_spike[i] and trending[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX negative and falling, volume spike, trending market
            elif (trix[i] < 0 and trix_falling[i] and volume_spike[i] and trending[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if TRIX turns negative or ranging market
            if (trix[i] <= 0) or ranging[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if TRIX turns positive or ranging market
            if (trix[i] >= 0) or ranging[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals