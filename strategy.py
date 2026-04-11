#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_trix_volume_regime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # TRIX calculation on daily close (period=12)
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean()
    trix = pd.Series(ema3).pct_change(periods=1) * 100  # TRIX as percentage
    trix_values = trix.values
    
    # Choppiness Index on daily (period=14) - regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    atr_1d = []
    for i in range(len(close_1d)):
        if i == 0:
            tr = high_1d[i] - low_1d[i]
        else:
            tr = max(high_1d[i] - low_1d[i], 
                     abs(high_1d[i] - close_1d[i-1]), 
                     abs(low_1d[i] - close_1d[i-1]))
        atr_1d.append(tr)
    
    atr_1d = np.array(atr_1d)
    atr_sum_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum()
    max_hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    min_ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum_14 / (max_hh - min_ll)) / np.log10(14)
    chop_values = chop.values
    
    # Volume confirmation: 4h volume > 1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # Align daily indicators to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix_values)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    for i in range(150, n):
        # Skip if any required data is invalid
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma_50[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 1.5 * vol_ma_50[i]
        
        # TRIX momentum signal
        trix_signal = trix_aligned[i]
        
        # Regime filter: Choppiness > 61.8 = ranging (mean revert), < 38.2 = trending
        chop_value = chop_aligned[i]
        is_ranging = chop_value > 61.8
        is_trending = chop_value < 38.2
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # In ranging market: mean reversion at TRIX extremes
        if is_ranging:
            if trix_signal < -1.5 and vol_confirm:  # Oversold
                enter_long = True
            elif trix_signal > 1.5 and vol_confirm:  # Overbought
                enter_short = True
        
        # In trending market: momentum continuation
        if is_trending:
            if trix_signal > 0 and vol_confirm:  # Bullish momentum
                enter_long = True
            elif trix_signal < 0 and vol_confirm:  # Bearish momentum
                enter_short = True
        
        # Exit conditions: opposite TRIX extreme or regime change
        exit_long = (trix_signal > 1.5) or (not is_ranging and not is_trending)  # Overbought or choppy
        exit_short = (trix_signal < -1.5) or (not is_ranging and not is_trending)  # Oversold or choppy
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 4h TRIX momentum with daily regime filter (Choppiness Index) and volume confirmation.
# In ranging markets (CHOP > 61.8): mean reversion at TRIX extremes (±1.5).
# In trending markets (CHOP < 38.2): momentum continuation (TRIX > 0 for long, < 0 for short).
# Uses volume confirmation (1.5x 50-period average) to filter false signals.
# Position size 0.25 to manage risk. Designed for both bull and bear markets via regime adaptation.
# Target: 20-30 trades per year (80-120 total over 4 years) to minimize fee drag.