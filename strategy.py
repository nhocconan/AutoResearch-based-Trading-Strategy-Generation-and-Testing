#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band mean reversion with weekly Bollinger Band width regime filter
# Uses Bollinger Band touches at 2 std dev for mean reversion entries
# Weekly Bollinger Band width percentile determines regime: narrow = mean reversion mode, wide = trend mode
# In narrow width regimes (low volatility), we fade BB touches; in wide width regimes (high volatility), we follow breakouts
# Designed for low trade frequency (target: 15-35 trades/year) to minimize fee drag
# Works in ranging markets via BB mean reversion and in trending markets via breakout continuation

name = "6h_bollinger_width_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for Bollinger Band width regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 6h Bollinger Bands (20, 2)
    bb_length = 20
    bb_mult = 2.0
    
    # Calculate BB middle (SMA)
    bb_middle = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean().values
    # Calculate BB standard deviation
    bb_std = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std().values
    # Calculate upper and lower bands
    bb_upper = bb_middle + bb_mult * bb_std
    bb_lower = bb_middle - bb_mult * bb_std
    
    # Weekly Bollinger Band width for regime detection
    close_1w = df_1w['close'].values
    bb_length_w = 20
    bb_mult_w = 2.0
    
    bb_middle_w = pd.Series(close_1w).rolling(window=bb_length_w, min_periods=bb_length_w).mean().values
    bb_std_w = pd.Series(close_1w).rolling(window=bb_length_w, min_periods=bb_length_w).std().values
    bb_width_w = (bb_middle_w + bb_mult_w * bb_std_w) - (bb_middle_w - bb_mult_w * bb_std_w)  # upper - lower = 2 * mult * std
    bb_width_w = 2 * bb_mult_w * bb_std_w
    
    # Percentile of BB width over 50 periods for regime classification
    bb_width_percentile = pd.Series(bb_width_w).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50, raw=False
    ).values
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1w, bb_width_percentile)
    
    signals = np.zeros(n)
    
    for i in range(bb_length, n):
        # Skip if required data not available
        if (np.isnan(bb_middle[i]) or np.isnan(bb_std[i]) or 
            np.isnan(bb_width_percentile_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime classification: narrow width = mean reversion mode, wide width = trend mode
        width_percentile = bb_width_percentile_aligned[i]
        is_narrow_width = width_percentile < 30  # Bottom 30% = low volatility ranging
        is_wide_width = width_percentile > 70    # Top 30% = high volatility trending
        
        # Price position relative to bands
        price_vs_upper = close[i] - bb_upper[i]
        price_vs_lower = bb_lower[i] - close[i]
        
        # Mean reversion signals in narrow width regimes (fade extremes)
        if is_narrow_width:
            if price_vs_lower > 0:  # Price below lower band -> long
                signals[i] = 0.25
            elif price_vs_upper > 0:  # Price above upper band -> short
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        # Breakout continuation signals in wide width regimes (follow breakouts)
        elif is_wide_width:
            if price_vs_upper > 0:  # Break above upper band -> long
                signals[i] = 0.25
            elif price_vs_lower > 0:  # Break below lower band -> short
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        # Neutral regime: no signal
        else:
            signals[i] = 0.0
    
    return signals