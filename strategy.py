#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout + 1d Volume Regime Filter
# Long when: BB(20,2) width < 20th percentile (squeeze) AND price breaks above upper band AND 1d volume > 1.5x 20-period MA
# Short when: BB(20,2) width < 20th percentile (squeeze) AND price breaks below lower band AND 1d volume > 1.5x 20-period MA
# Exit when: price returns to middle band (mean reversion) OR squeeze breaks without follow-through
# Uses volatility contraction/expansion principle - works in both bull/bear markets by capturing breakouts from low volatility
# Timeframe: 6h, HTF: 1d for volume regime filter. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_BBSqueeze_Breakout_1dVolRegime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands on 6h
    if len(close) >= 20:
        sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
        std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
        upper_band = sma_20 + (2 * std_20)
        lower_band = sma_20 - (2 * std_20)
        bb_width = (upper_band - lower_band) / sma_20  # normalized width
    else:
        sma_20 = np.full(n, np.nan)
        upper_band = np.full(n, np.nan)
        lower_band = np.full(n, np.nan)
        bb_width = np.full(n, np.nan)
    
    # Calculate BB width percentile (20th) for squeeze detection
    bb_width_pct_20 = np.full(n, np.nan)
    for i in range(50, n):  # need sufficient history for percentile
        window = bb_width[max(0, i-50):i+1]  # 50-bar lookback for percentile
        valid_window = window[~np.isnan(window)]
        if len(valid_window) >= 10:
            bb_width_pct_20[i] = np.percentile(valid_window, 20)
    
    # Squeeze condition: BB width below 20th percentile
    squeeze = bb_width < bb_width_pct_20
    
    # Breakout conditions
    breakout_up = (close > upper_band) & squeeze
    breakout_down = (close < lower_band) & squeeze
    
    # Get 1d data ONCE before loop for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need sufficient data
        return np.zeros(n)
    
    # Calculate 1d volume MA
    vol_1d = df_1d['volume'].values
    if len(vol_1d) >= 20:
        vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        volume_regime = vol_1d > (1.5 * vol_ma_20_1d)  # high volume regime
    else:
        volume_regime = np.zeros(len(df_1d), dtype=bool)
    
    # Align 1d volume regime to 6h timeframe
    volume_regime_aligned = align_htf_to_ltf(prices, df_1d, volume_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or 
            np.isnan(volume_regime_aligned[i]) or np.isnan(sma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout with volume confirmation
            if breakout_up[i] and volume_regime_aligned[i] == 1.0:
                signals[i] = 0.25
                position = 1
            # Short breakout with volume confirmation
            elif breakout_down[i] and volume_regime_aligned[i] == 1.0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle band OR squeeze breaks without follow-through
            if close[i] <= sma_20[i] or not squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle band OR squeeze breaks without follow-through
            if close[i] >= sma_20[i] or not squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals