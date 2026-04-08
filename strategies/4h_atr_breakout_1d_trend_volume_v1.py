#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h ATR Breakout with 1d Trend Filter and Volume Confirmation
# Hypothesis: ATR-based breakouts capture volatility expansion moves.
# Combined with 1d EMA50 trend filter to avoid counter-trend trades.
# Volume confirmation ensures breakouts have institutional participation.
# Works in both bull and bear markets by only taking trades aligned with higher timeframe trend.
# Targets 20-30 trades/year with disciplined entries to avoid overtrading.

name = "4h_atr_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h ATR(14) for volatility measurement
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 20-period SMA for volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup for ATR and volume SMA
        # Skip if required data not available
        if (np.isnan(ema50_4h[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below 4h EMA50 OR ATR-based trailing stop
            if close[i] < ema50_4h[i] or close[i] < high[max(0, i-3):i+1].max() - 2.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above 4h EMA50 OR ATR-based trailing stop
            if close[i] > ema50_4h[i] or close[i] > low[max(0, i-3):i+1].min() + 2.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above recent high + volume confirmation + uptrend
            if (close[i] > high[max(0, i-5):i].max() and 
                vol_confirm and 
                close[i] > ema50_4h[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below recent low + volume confirmation + downtrend
            elif (close[i] < low[max(0, i-5):i].min() and 
                  vol_confirm and 
                  close[i] < ema50_4h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals