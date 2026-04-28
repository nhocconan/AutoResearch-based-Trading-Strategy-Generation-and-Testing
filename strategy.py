#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX + Volume Spike + Choppiness Regime Filter
# TRIX (triple exponential smoothed MACD) catches momentum with less whipsaw.
# Long: TRIX crosses above signal line + volume spike + choppy regime (mean reversion)
# Short: TRIX crosses below signal line + volume spike + choppy regime
# Uses choppy regime (CHOP > 61.8) to avoid strong trends where TRIX whipsaws.
# Volume confirmation ensures breakouts have conviction.
# Discrete sizing 0.25 limits drawdown and fee churn.
# Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_TRIX_VolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate TRIX (15,9,9)
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15)
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3.pct_change())  # Percentage change
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = trix.ewm(span=9, adjust=False, min_periods=9).mean()
    trix_hist = trix - trix_signal  # Histogram for crossover detection
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    # Choppiness Index regime filter (14-period)
    # CHOP = 100 * log10(sum(ATR(1)) / (n * log(n))) / log10(n)
    # Where ATR(1) = TR (true range) for each period
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # Align with index
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).sum()
    n_val = 14
    chop = 100 * np.log10(atr1 / (n_val * np.log10(n_val))) / np.log10(n_val)
    chop_values = chop.values
    chop_regime = chop_values > 61.8  # Choppy/range regime (mean reversion favorable)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure sufficient history for TRIX and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_hist[i]) or np.isnan(trix_hist[i-1]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Choppiness regime filter
        in_chop = chop_regime[i]
        
        # TRIX crossovers
        trix_cross_up = trix_hist[i] > 0 and trix_hist[i-1] <= 0
        trix_cross_down = trix_hist[i] < 0 and trix_hist[i-1] >= 0
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: TRIX bullish crossover + volume confirm + choppy regime
            if trix_cross_up and vol_confirm and in_chop:
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX bearish crossover + volume confirm + choppy regime
            elif trix_cross_down and vol_confirm and in_chop:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on TRIX bearish crossover
            if trix_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on TRIX bullish crossover
            if trix_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals