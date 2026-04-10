#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Mean Reversion with 1d ATR regime filter
# - Primary: 6h timeframe for balanced trade frequency and reduced fee drag
# - HTF: 1d for ATR-based volatility regime (high ATR = trending, low ATR = ranging)
# - Logic: In low volatility regimes (ATR < 30th percentile), fade extreme Williams %R readings
#          Long: Williams %R < -80 (oversold) + low ATR regime
#          Short: Williams %R > -20 (overbought) + low ATR regime
#          Exit: Williams %R returns to -50 level (mean reversion) or ATR regime shifts to high
# - Position sizing: 0.25 (discrete level)
# - Target: 80-180 total trades over 4 years (20-45/year) - within 6h sweet spot
# - Works in bull/bear: Mean reversion effective in ranging markets (2025), volatility filter avoids false signals in trends

name = "6h_1d_williamsr_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 6h Williams %R (14-period)
    highest_high_6h = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_6h = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_6h - close_6h) / (highest_high_6h - lowest_low_6h)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_6h - lowest_low_6h) == 0, -50, williams_r)
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR percentile rank (using 30-day lookback) for regime filter
    atr_percentile = pd.Series(atr_1d).rolling(window=30, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or 
            np.isnan(atr_percentile_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime condition: low volatility (ATR < 30th percentile) = ranging market
        low_vol_regime = atr_percentile_aligned[i] < 30
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R deeply oversold + low volatility regime
            if (williams_r[i] < -80 and low_vol_regime):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R deeply overbought + low volatility regime
            elif (williams_r[i] > -20 and low_vol_regime):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Williams %R returns to -50 (mean reversion)
            # 2. Volatility regime shifts to high (ATR >= 30th percentile) = potential trend start
            
            if position == 1:  # Long position
                exit_condition = (
                    williams_r[i] > -50 or  # Mean reversion to midpoint
                    atr_percentile_aligned[i] >= 30  # Regime shift to higher volatility
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    williams_r[i] < -50 or  # Mean reversion to midpoint
                    atr_percentile_aligned[i] >= 30  # Regime shift to higher volatility
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals