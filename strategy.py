#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band breakout with volume confirmation and 1d trend filter.
# Uses Bollinger Bands (20, 2) for mean reversion in ranging markets and breakouts in trending markets.
# Volume confirmation ensures institutional participation. Works in both bull and bear markets by
# adapting to volatility regime via Bollinger Band width.
name = "4h_Bollinger_Band_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Bollinger Bands (20, 2) on 4h
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean()
    bb_std = close_series.rolling(window=20, min_periods=20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_upper = bb_upper.values
    bb_lower = bb_lower.values
    
    # Volume spike detection (20-period MA)
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma20[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: Price breaks above upper BB with volume and 1d uptrend
            if close[i] > bb_upper[i] and vol_ok and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower BB with volume and 1d downtrend
            elif close[i] < bb_lower[i] and vol_ok and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below middle BB
            if close[i] < bb_middle.iloc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above middle BB
            if close[i] > bb_middle.iloc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals