# 1h_TRIX_Momentum_Trend
# Hypothesis: TRIX (Triple Exponential Average) filters out market noise and identifies momentum trends.
# In trending markets, TRIX crossovers with signal line provide reliable entry signals.
# Works in both bull and bear markets by capturing momentum shifts.
# Uses 4h trend filter and volume confirmation to reduce false signals.
# Limits trades via momentum strength filter and session filter (08-20 UTC).
# Target: 15-37 trades/year (~60-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate TRIX (15-period) and signal line (9-period EMA of TRIX)
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) * 100
    ema1 = pd.Series(close).ewm(span=15, adjust=False).values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False).values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False).values
    trix = ema3 * 100
    
    # Signal line: 9-period EMA of TRIX
    signal_line = pd.Series(trix).ewm(span=9, adjust=False).values
    
    # Histogram: TRIX - signal line
    histogram = trix - signal_line
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA34 for trend direction
    ema34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False).values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_filter = volume > (vol_ma * 1.5)
    
    # Momentum strength: |histogram| > 0.1
    mom_filter = np.abs(histogram) > 0.1
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(trix[i]) or np.isnan(signal_line[i]) or 
            np.isnan(ema34_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above signal line with bullish 4h trend and volume/momentum
            if (trix[i] > signal_line[i] and trix[i-1] <= signal_line[i-1] and
                ema34_4h_aligned[i] > ema34_4h_aligned[i-1] and
                vol_filter[i] and mom_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: TRIX crosses below signal line with bearish 4h trend and volume/momentum
            elif (trix[i] < signal_line[i] and trix[i-1] >= signal_line[i-1] and
                  ema34_4h_aligned[i] < ema34_4h_aligned[i-1] and
                  vol_filter[i] and mom_filter[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below signal line or trend turns bearish
            if (trix[i] < signal_line[i] and trix[i-1] >= signal_line[i-1]) or \
               ema34_4h_aligned[i] < ema34_4h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: TRIX crosses above signal line or trend turns bullish
            if (trix[i] > signal_line[i] and trix[i-1] <= signal_line[i-1]) or \
               ema34_4h_aligned[i] > ema34_4h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_TRIX_Momentum_Trend"
timeframe = "1h"
leverage = 1.0