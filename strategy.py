#!/usr/bin/env python3
"""
4h_1d_TRIX_Volume_Spike_Regime_v1
Hypothesis: Uses TRIX (1-period ROC of EMA) on 4h with volume spike and 1-day Choppiness regime filter.
TRIX > 0 indicates bullish momentum, TRIX < 0 bearish. Volume spike confirms breakout strength.
Choppiness regime (from daily) filters: CHOP > 61.8 = range (mean revert at extremes), CHOP < 38.2 = trend (follow TRIX).
Designed for low trade frequency (<30/year) to avoid fee drag, works in bull/bear via regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_TRIX_Volume_Spike_Regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Choppiness index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate TRIX on 4h: EMA of EMA of EMA of close, then 1-period ROC
    close_s = pd.Series(close)
    ema1 = close_s.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.pct_change(1) * 100  # 1-period ROC as percentage
    trix = trix.values
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR(14) sum of TR
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(atr_sum / (hh - ll)) / log10(14)
    chop = 100 * np.log10(atr_14 / (hh_14 - ll_14 + 1e-10)) / np.log10(14)
    chop = chop.values
    
    # Align Choppiness to 4h timeframe (wait for daily close)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(trix[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average (strong spike)
        volume_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        # TRIX signals
        trix_bullish = trix[i] > 0
        trix_bearish = trix[i] < 0
        
        # Choppiness regime: >61.8 = range, <38.2 = trend
        chop_value = chop_aligned[i]
        is_range = chop_value > 61.8
        is_trend = chop_value < 38.2
        
        # Entry logic: adapt to regime
        if is_range:
            # In range: mean reversion at extremes (TRIX near zero)
            long_entry = volume_filter and trix[i] < -0.1 and trix[i] > -0.5  # Oversold bounce
            short_entry = volume_filter and trix[i] > 0.1 and trix[i] < 0.5   # Overbought pullback
        else:  # trending or neutral chop
            # In trend: follow TRIX direction
            long_entry = volume_filter and trix_bullish and trix[i] > 0.1
            short_entry = volume_filter and trix_bearish and trix[i] < -0.1
        
        # Exit conditions: opposite TRIX signal or volume dry-up
        long_exit = (trix[i] < -0.1) or (volume[i] < 0.5 * vol_ma_20[i])
        short_exit = (trix[i] > 0.1) or (volume[i] < 0.5 * vol_ma_20[i])
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals