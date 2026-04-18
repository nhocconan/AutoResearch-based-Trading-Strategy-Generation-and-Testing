#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX momentum with volume spike confirmation and 1d EMA50 trend filter.
# TRIX (12-period) filters noise and captures momentum. 
# Entry: TRIX crosses above/below signal line + volume spike + trend filter.
# Exit: TRIX crosses back or trend reversal.
# Uses 1d EMA50 to ensure trades align with daily trend, reducing whipsaw.
# Target: 20-50 trades/year to minimize fee drag and ensure robustness.
name = "4h_TRIX_Momentum_VolumeTrend"
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
    
    # Get 1d data for EMA50 trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate TRIX: triple EMA of ROC
    # ROC = (close - close_prev) / close_prev * 100
    close_series = pd.Series(close)
    roc = close_series.pct_change() * 100
    # Triple EMA of ROC
    ema1 = roc.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.values
    # Signal line: EMA of TRIX
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(trix[i]) or np.isnan(trix_signal[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # TRIX signals: bullish when TRIX > signal line, bearish when TRIX < signal line
        trix_bullish = trix[i] > trix_signal[i]
        trix_bearish = trix[i] < trix_signal[i]
        
        if position == 0:
            # Long: TRIX bullish crossover + volume spike + uptrend
            if trix_bullish and trix[i-1] <= trix_signal[i-1] and volume_spike[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: TRIX bearish crossover + volume spike + downtrend
            elif trix_bearish and trix[i-1] >= trix_signal[i-1] and volume_spike[i] and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TRIX turns bearish OR trend reverses
            if not trix_bullish or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX turns bullish OR trend reverses
            if trix_bullish or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals