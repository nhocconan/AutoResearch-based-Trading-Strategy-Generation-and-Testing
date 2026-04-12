#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_12h_trix_volume_regime_v1
# Uses TRIX momentum on 12h for trend direction, volume confirmation, and choppiness regime filter.
# In trending markets (CHOP < 50), we follow TRIX crosses; in choppy markets (CHOP >= 50), we avoid trades.
# This reduces whipsaws in sideways markets while capturing trends. Target: 20-40 trades/year per symbol.
name = "4h_12h_trix_volume_regime_v1"
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
    
    # Get 12h data for TRIX calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate TRIX on 12h close: TRIX = EMA(EMA(EMA(close, 12), 12), 12)
    close_12h = df_12h['close'].values
    ema1 = pd.Series(close_12h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix_raw = pd.Series(ema3).pct_change() * 100  # percentage change
    trix = trix_raw.values
    
    # Align TRIX to 4h timeframe (wait for completed 12h bar)
    trix_aligned = align_htf_to_ltf(prices, df_12h, trix)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    # Chop regime filter: avoid choppy markets (CHOP >= 50)
    # Calculate CHOP using 14-period ATR and highest/lowest
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero or invalid values
    denominator = atr * np.sqrt(14)
    chop = np.where(denominator > 0, 100 * np.log10((highest_high - lowest_low) / denominator) / np.log10(14), 100)
    chop_filter = chop < 50  # trending market (lower threshold for more signals)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after warmup
        # Skip if TRIX not ready
        if np.isnan(trix_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Check volume and chop filters
        if not (vol_confirm[i] and chop_filter[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: TRIX crosses above zero with volume
        if i > 0 and not np.isnan(trix_aligned[i-1]) and trix_aligned[i-1] <= 0 and trix_aligned[i] > 0 and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: TRIX crosses below zero with volume
        elif i > 0 and not np.isnan(trix_aligned[i-1]) and trix_aligned[i-1] >= 0 and trix_aligned[i] < 0 and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite TRIX cross
        elif i > 0 and not np.isnan(trix_aligned[i-1]) and trix_aligned[i-1] < 0 and trix_aligned[i] >= 0 and position == 1:
            position = 0
            signals[i] = 0.0
        elif i > 0 and not np.isnan(trix_aligned[i-1]) and trix_aligned[i-1] > 0 and trix_aligned[i] <= 0 and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals