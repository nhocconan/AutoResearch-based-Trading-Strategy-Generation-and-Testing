#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR volume regime and 4h EMA50 trend filter.
Long when price breaks above 20-period Donchian high with 1d ATR > 1.5x 20-day ATR average and price > 4h EMA50.
Short when price breaks below 20-period Donchian low with 1d ATR > 1.5x 20-day ATR average and price < 4h EMA50.
Exit when price returns to 20-period Donchian midpoint or reverses with volume confirmation.
Uses 1d for ATR-based volatility regime filter, 4h for execution and trend filter.
Designed to capture volatility expansion breakouts in both bull and bear markets.
Volatility regime filter ensures trades only occur during periods of higher volatility, reducing whipsaws in low-volatility ranging markets.
Target: 20-40 trades/year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR-based volatility regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d True Range and ATR(20)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR MA20 for regime filter
    atr_ma_20_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Calculate 4h EMA50 for trend filter
    close_series = pd.Series(close)
    ema50_4h = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA50 and ATR calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or 
            np.isnan(ema50_4h[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume/volatility regime: current 1d ATR > 1.5x 20-day ATR average (expanding volatility)
        vol_regime = atr_1d_aligned[i] > 1.5 * atr_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volatility regime and uptrend (price > EMA50)
            if (close[i] > donch_high[i] and 
                vol_regime and 
                close[i] > ema50_4h[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volatility regime and downtrend (price < EMA50)
            elif (close[i] < donch_low[i] and 
                  vol_regime and 
                  close[i] < ema50_4h[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below Donchian midpoint OR breaks below Donchian low with volatility (reversal)
            if (close[i] <= donch_mid[i] or 
                (close[i] < donch_low[i] and vol_regime)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above Donchian midpoint OR breaks above Donchian high with volatility (reversal)
            if (close[i] >= donch_mid[i] or 
                (close[i] > donch_high[i] and vol_regime)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATRVolRegime_EMA50_Trend"
timeframe = "4h"
leverage = 1.0