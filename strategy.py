#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d ATR-based volatility filter + volume confirmation
# Williams %R identifies overbought/oversold conditions; in high volatility regimes (ATR ratio > 1.2),
# we fade extremes with volume confirmation. In low volatility (ATR ratio <= 1.2), we follow momentum.
# Uses discrete position sizing 0.25 to target ~12-37 trades/year and minimize fee drag.
# Works in bull/bear markets: mean reversion in high vol, momentum in low vol.

name = "12h_1d_williamsr_vol_filter_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(14) for volatility normalization
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Calculate 1d ATR ratio (current ATR / 20-period average ATR) for volatility regime
    atr_avg_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio_1d = np.where(atr_avg_1d > 0, atr_1d / atr_avg_1d, np.nan)
    
    # Calculate 1d Williams %R(14) based on prior day to avoid look-ahead
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = -100 * (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d)
    
    # Calculate 1d average volume (20-period)
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Pre-compute volume confirmation array
    volume_confirmed = volume > 1.5 * avg_volume_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(williams_r_1d_aligned[i]) or
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter
        high_vol_regime = atr_ratio_1d_aligned[i] > 1.2
        low_vol_regime = atr_ratio_1d_aligned[i] <= 1.2
        
        if position == 1:  # Long position
            if high_vol_regime:
                # Exit long if Williams %R rises above -20 (overbought) or volatility drops
                if williams_r_1d_aligned[i] > -20 or low_vol_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif low_vol_regime:
                # Exit long if Williams %R falls below -80 (oversold) or volatility rises
                if williams_r_1d_aligned[i] < -80 or high_vol_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if high_vol_regime:
                # Exit short if Williams %R falls below -80 (oversold) or volatility drops
                if williams_r_1d_aligned[i] < -80 or low_vol_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif low_vol_regime:
                # Exit short if Williams %R rises above -20 (overbought) or volatility rises
                if williams_r_1d_aligned[i] > -20 or high_vol_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if high_vol_regime:
                # Enter long near oversold (-80) with volume confirmation
                if williams_r_1d_aligned[i] <= -80 and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short near overbought (-20) with volume confirmation
                elif williams_r_1d_aligned[i] >= -20 and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
            elif low_vol_regime:
                # Enter long on momentum up (Williams %R rising from oversold)
                if williams_r_1d_aligned[i] > williams_r_1d_aligned[i-1] and williams_r_1d_aligned[i] < -50 and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short on momentum down (Williams %R falling from overbought)
                elif williams_r_1d_aligned[i] < williams_r_1d_aligned[i-1] and williams_r_1d_aligned[i] > -50 and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals