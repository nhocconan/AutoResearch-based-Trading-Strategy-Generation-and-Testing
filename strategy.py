#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Bands width regime + mean reversion with volume confirmation.
# Uses daily Bollinger Band width percentile to filter regimes:
# - When BB width < 20th percentile (low volatility squeeze): mean reversion at ±1.5σ
# - When BB width > 80th percentile (high volatility expansion): trend following with breakout
# Volume filter ensures institutional participation. Designed for 15-35 trades/year on 6h.
# Works in both bull/bear markets by adapting to volatility regimes.

name = "6h_1d_bb_width_regime_meanrev_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # Calculate 20-period percentile rank of BB width (using expanding window for no look-ahead)
    bb_width_percentile = np.full_like(bb_width, np.nan)
    for i in range(20, len(bb_width)):
        bb_width_percentile[i] = (bb_width[:i+1] <= bb_width[i]).mean() * 100
    
    # Align BB width percentile to 6h timeframe
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    # Calculate 6h Bollinger Bands for entry signals
    sma_20_6h = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20_6h = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb_6h = sma_20_6h + 2 * std_20_6h
    lower_bb_6h = sma_20_6h - 2 * std_20_6h
    
    # Calculate 6h volume moving average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(bb_width_percentile_aligned[i]) or 
            np.isnan(sma_20_6h[i]) or np.isnan(std_20_6h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * 20-period average volume
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Regime determination based on daily BB width percentile
        is_low_vol = bb_width_percentile_aligned[i] < 20  # Squeeze regime
        is_high_vol = bb_width_percentile_aligned[i] > 80  # Expansion regime
        
        # Mean reversion signals (in low volatility squeeze)
        mean_rev_long = (close[i] <= lower_bb_6h[i]) and vol_filter and is_low_vol
        mean_rev_short = (close[i] >= upper_bb_6h[i]) and vol_filter and is_low_vol
        
        # Trend following signals (in high volatility expansion)
        trend_long = (close[i] > upper_bb_6h[i]) and vol_filter and is_high_vol
        trend_short = (close[i] < lower_bb_6h[i]) and vol_filter and is_high_vol
        
        # Exit conditions: opposite signal or volatility regime change
        exit_long = (close[i] >= sma_20_6h[i]) or (is_low_vol and position == 1) or (is_high_vol and position == 1 and close[i] < sma_20_6h[i])
        exit_short = (close[i] <= sma_20_6h[i]) or (is_low_vol and position == -1) or (is_high_vol and position == -1 and close[i] > sma_20_6h[i])
        
        # Priority: entry > exit > hold
        if (mean_rev_long or trend_long) and position != 1:
            position = 1
            signals[i] = 0.25
        elif (mean_rev_short or trend_short) and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals