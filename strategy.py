#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d volume spike and ATR-based regime filter
# In low volatility regimes (ATR ratio < 0.8): breakout above/below Donchian(20) levels with volume confirmation
# In high volatility regimes (ATR ratio > 1.2): mean reversion at Donchian midpoint with volume confirmation
# Uses discrete position sizing 0.25 to limit trades to ~12-37/year and reduce fee drag
# Works in bull/bear markets: breakout catches trends, mean reversion captures volatility mean reversion

name = "12h_1d_donchian_breakout_volume_atr_v1"
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
    
    # Calculate 1d ATR(14)
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
    
    # Calculate 1d ATR ratio (current ATR / 20-period average ATR) for regime detection
    atr_ma_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio = np.where(atr_ma_20 > 0, atr_1d / atr_ma_20, np.nan)
    
    # Calculate 1d average volume (20-period)
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Donchian channels (20-period) - using prior day to avoid look-ahead
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high_20 + donchian_low_20) / 2.0
    
    # Align 1d indicators to 12h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Pre-compute volume confirmation array
    volume_confirmed = volume > 1.5 * avg_volume_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(donchian_high_20_aligned[i]) or
            np.isnan(donchian_low_20_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter based on ATR ratio
        low_vol_regime = atr_ratio_aligned[i] < 0.8
        high_vol_regime = atr_ratio_aligned[i] > 1.2
        
        if position == 1:  # Long position
            if low_vol_regime:
                # Exit long if price breaks below Donchian low or we enter high vol regime
                if close[i] < donchian_low_20_aligned[i] or high_vol_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif high_vol_regime:
                # Exit long if price rises above midpoint or drops below low
                if close[i] > donchian_mid_aligned[i] or close[i] < donchian_low_20_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if low_vol_regime:
                # Exit short if price breaks above Donchian high or we enter high vol regime
                if close[i] > donchian_high_20_aligned[i] or high_vol_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif high_vol_regime:
                # Exit short if price drops below midpoint or rises above high
                if close[i] < donchian_mid_aligned[i] or close[i] > donchian_high_20_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if low_vol_regime:
                # Enter long on breakout above Donchian high with volume confirmation
                if close[i] > donchian_high_20_aligned[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short on breakout below Donchian low with volume confirmation
                elif close[i] < donchian_low_20_aligned[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
            elif high_vol_regime:
                # Mean reversion: buy near low, sell near high
                if close[i] <= donchian_low_20_aligned[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= donchian_high_20_aligned[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals