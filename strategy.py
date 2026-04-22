# 12h_TRIX_Volume_Spike_1dTrend
# Hypothesis: TRIX momentum reversal on 12h combined with volume spike and 1d trend filter.
# TRIX (12-period) crosses above/below zero with volume > 2x 20-period average and 1d EMA50 trend alignment.
# Works in bull and bear by following 1d trend. Target 15-30 trades/year.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TRIX (12, 12, 9) on 12h close
    ema1 = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema2 = ema1.ewm(span=12, adjust=False).mean()
    ema3 = ema2.ewm(span=12, adjust=False).mean()
    trix = 100 * (ema3 / ema3.shift(1) - 1)
    trix_signal = trix.ewm(span=9, adjust=False).mean()
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_ltf_to_htf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(trix_signal.iloc[i]) or np.isnan(trix_signal.iloc[i-1]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema50_1d_aligned[i])):
            continue
            
        # TRIX crosses zero with volume spike and 1d trend alignment
        trix_cross_up = trix_signal.iloc[i-1] <= 0 and trix_signal.iloc[i] > 0
        trix_cross_down = trix_signal.iloc[i-1] >= 0 and trix_signal.iloc[i] < 0
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if trix_cross_up and vol_spike and ema50_1d_aligned[i] > ema50_1d_aligned[i-1]:
            signals[i] = 0.25
        elif trix_cross_down and vol_spike and ema50_1d_aligned[i] < ema50_1d_aligned[i-1]:
            signals[i] = -0.25
    
    return signals

name = "12h_TRIX_Volume_Spike_1dTrend"
timeframe = "12h"
leverage = 1.0