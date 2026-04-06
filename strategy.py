#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d/1w regime filter
# Williams Alligator consists of three SMAs (Jaw=13, Teeth=8, Lips=5) smoothed
# Jaw (blue): 13-period SMA smoothed by 8 bars
# Teeth (red): 8-period SMA smoothed by 5 bars
# Lips (green): 5-period SMA smoothed by 3 bars
# In uptrend: Lips > Teeth > Jaw; in downtrend: Lips < Teeth < Jaw
# When lines intertwine (no clear order), market is ranging
# Long when Lips > Teeth > Jaw with 1d/1w uptrend and volume confirmation
# Short when Lips < Teeth < Jaw with 1d/1w downtrend and volume confirmation
# Uses volume confirmation to avoid false signals
# Target: 50-150 total trades over 4 years with controlled risk in both bull and bear markets

name = "12h_williams_alligator_1d_1w_regime_v1"
timeframe = "12h"
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
    
    # 1d and 1w data for regime filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    
    # 1d and 1w EMA(50) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Williams Alligator components (using close prices)
    # Jaw (blue): 13-period SMA smoothed by 8 bars
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(jaw_raw).rolling(window=8, min_periods=8).mean().values
    
    # Teeth (red): 8-period SMA smoothed by 5 bars
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth_raw).rolling(window=5, min_periods=5).mean().values
    
    # Lips (green): 5-period SMA smoothed by 3 bars
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips_raw).rolling(window=3, min_periods=3).mean().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):  # Start after warmup for Alligator
        # Skip if required data not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(ema_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation using price range
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Alligator lines reverse or 1d/1w trend turns down
            elif (lips[i] < teeth[i] or teeth[i] < jaw[i]) or \
                 (ema_1d_aligned[i] < ema_1d_aligned[i-1] and ema_1w_aligned[i] < ema_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            if close[i] > entry_price + 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Alligator lines reverse or 1d/1w trend turns up
            elif (lips[i] > teeth[i] or teeth[i] > jaw[i]) or \
                 (ema_1d_aligned[i] > ema_1d_aligned[i-1] and ema_1w_aligned[i] > ema_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if vol_filter[i]:
                # Long when Lips > Teeth > Jaw with 1d/1w uptrend
                if lips[i] > teeth[i] and teeth[i] > jaw[i] and \
                   ema_1d_aligned[i] > ema_1d_aligned[i-1] and ema_1w_aligned[i] > ema_1w_aligned[i-1]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short when Lips < Teeth < Jaw with 1d/1w downtrend
                elif lips[i] < teeth[i] and teeth[i] < jaw[i] and \
                     ema_1d_aligned[i] < ema_1d_aligned[i-1] and ema_1w_aligned[i] < ema_1w_aligned[i-1]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals