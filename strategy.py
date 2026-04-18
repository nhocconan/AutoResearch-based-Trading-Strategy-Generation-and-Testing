# 4h_TRIX_1dVix_Filtered_Signal_v1
# Hypothesis: Use TRIX(12,9) on 4h for momentum signal, combined with 1d VIX-like volatility index to filter noise.
# TRIX > 0 indicates bullish momentum, TRIX < 0 bearish. VIX filter enters only when volatility is elevated (>1.5x avg)
# to capture breakouts during high volatility periods. Works in both bull and bear by trading volatility expansion.
# Target: 20-40 trades/year per symbol to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TRIX calculation on 4h close
    ema1 = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False).mean()
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False).mean()
    trix_raw = pd.Series(ema3).pct_change(periods=1) * 100
    trix = trix_raw.values
    
    # 1d volatility index (VIX-like): ATR(14) normalized by SMA(20)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    vix_proxy = atr_14 / sma_20
    
    # Align TRIX and VIX proxy to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    vix_aligned = align_htf_to_ltf(prices, df_1d, vix_proxy)
    
    # VIX filter: elevated volatility (>1.5x 50-period average)
    vix_ma = pd.Series(vix_aligned).rolling(window=50, min_periods=50).mean().values
    volatility_filter = vix_aligned > 1.5 * vix_ma
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(trix_aligned[i]) or np.isnan(vix_aligned[i]) or np.isnan(vix_ma[i]):
            signals[i] = 0.0
            continue
            
        # Long: TRIX crosses above 0 with elevated volatility
        if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and volatility_filter[i]:
            signals[i] = 0.25
            position = 1
        # Short: TRIX crosses below 0 with elevated volatility
        elif trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and volatility_filter[i]:
            signals[i] = -0.25
            position = -1
        # Exit: TRIX crosses zero in opposite direction
        elif position == 1 and trix_aligned[i] < 0:
            signals[i] = 0.0
            position = 0
        elif position == -1 and trix_aligned[i] > 0:
            signals[i] = 0.0
            position = 0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "4h_TRIX_1dVix_Filtered_Signal_v1"
timeframe = "4h"
leverage = 1.0