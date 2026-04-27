#!/usr/bin/env python3
"""
1h_Triple_Phase_Confirmation
Hypothesis: Use 1d trend (EMA200) as regime filter, 4h momentum (MACD histogram) for direction, and 1h volume spike for entry timing. 
Avoids overtrading by requiring alignment across three timeframes. Works in bull/bear via trend filter.
Target: 15-35 trades/year (~60-140 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Get 1d data for trend regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 1d trend: EMA200
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Get 4h data for momentum signal
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # 4h MACD for momentum direction
    close_4h = df_4h['close'].values
    ema12 = pd.Series(close_4h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close_4h).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    macd_hist_aligned = align_htf_to_ltf(prices, df_4h, macd_hist)
    
    # 1h volume spike for entry confirmation
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = max(200, 34, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(macd_hist_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter: price vs 1d EMA200
        regime_bull = close[i] > ema200_1d_aligned[i]
        regime_bear = close[i] < ema200_1d_aligned[i]
        
        # Momentum: 4h MACD histogram
        mom_bull = macd_hist_aligned[i] > 0
        mom_bear = macd_hist_aligned[i] < 0
        
        # Entry timing: 1h volume spike
        vol_spike_now = vol_spike[i]
        
        if position == 0:
            # Long: bullish regime + bullish momentum + volume spike
            if regime_bull and mom_bull and vol_spike_now:
                signals[i] = size
                position = 1
            # Short: bearish regime + bearish momentum + volume spike
            elif regime_bear and mom_bear and vol_spike_now:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: regime turns bearish or momentum turns bearish
            if not regime_bull or not mom_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: regime turns bullish or momentum turns bullish
            if not regime_bear or not mom_bear:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Triple_Phase_Confirmation"
timeframe = "1h"
leverage = 1.0