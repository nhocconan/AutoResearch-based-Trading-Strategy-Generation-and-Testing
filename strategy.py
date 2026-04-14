#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1-week CCI for trend strength and 1-week Williams %R for momentum extremes.
# Long when weekly CCI > 100 (strong uptrend) and weekly Williams %R < -80 (oversold bounce).
# Short when weekly CCI < -100 (strong downtrend) and weekly Williams %R > -20 (overbought bounce).
# Exit when CCI returns to zero line or Williams %R reaches neutral territory (-50).
# Uses weekly momentum/oscillator extremes to catch reversals in both bull and bear markets,
# with CCI filtering for strong trends to avoid whipsaws in ranging conditions.
# Target: 10-25 trades/year per symbol (40-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load weekly data ONCE for CCI and Williams %R
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need enough for CCI(20) and Williams %R(14)
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate CCI (20)
    tp_1w = (high_1w + low_1w + close_1w) / 3.0
    sma_tp = pd.Series(tp_1w).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp_1w).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (tp_1w - sma_tp) / (0.015 * mad)
    
    # Calculate Williams %R (14)
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    willr = -100 * (highest_high - close_1w) / (highest_high - lowest_low)
    
    # Align indicators to daily timeframe
    cci_aligned = align_htf_to_ltf(prices, df_1w, cci)
    willr_aligned = align_htf_to_ltf(prices, df_1w, willr)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 34  # Need CCI and Williams %R periods
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(cci_aligned[i]) or 
            np.isnan(willr_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for momentum extremes in strong trends
            # Long: strong uptrend (CCI > 100) + oversold (Williams %R < -80)
            if (cci_aligned[i] > 100 and 
                willr_aligned[i] < -80):
                position = 1
                signals[i] = position_size
            # Short: strong downtrend (CCI < -100) + overbought (Williams %R > -20)
            elif (cci_aligned[i] < -100 and 
                  willr_aligned[i] > -20):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend weakening or momentum normalizing
            if (cci_aligned[i] <= 0 or 
                willr_aligned[i] >= -50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: trend weakening or momentum normalizing
            if (cci_aligned[i] >= 0 or 
                willr_aligned[i] <= -50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_CCI_WilliamsR_MomentumExtremes_v1"
timeframe = "1d"
leverage = 1.0