#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian(20) breakout with 1d volume confirmation and ATR-based regime filter
# Hypothesis: Breakouts work best with volume confirmation and in higher volatility regimes (ATR above median)
# During low volatility (ATR below median), avoid trades to reduce whipsaw
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 20-50 trades/year.
name = "4h_donchian20_1d_volume_atrregime_v1"
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
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily 20-period volume moving average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate ATR(14) for volatility regime filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR 50-period median for regime filter (avoid low volatility chop)
    atr_median = pd.Series(atr).rolling(window=50, min_periods=50).median().values
    
    # Calculate Donchian channels (20-period high/low)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(atr_median[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > daily average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        # Regime filter: only trade when ATR is above its 50-period median (avoid low volatility chop)
        regime_filter = atr[i] > atr_median[i]
        
        if position == 1:  # Long position
            # Exit: price touches opposite band OR regime changes to low volatility
            if close[i] <= lowest_low[i] or not regime_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price touches opposite band OR regime changes to low volatility
            if close[i] >= highest_high[i] or not regime_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price breaks above upper band + volume confirmation + regime filter
            if close[i] > highest_high[i] and vol_confirm and regime_filter:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below lower band + volume confirmation + regime filter
            elif close[i] < lowest_low[i] and vol_confirm and regime_filter:
                position = -1
                signals[i] = -0.25
    
    return signals