#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + chop regime filter
# - Primary signal: Price breaks above/below 20-period Donchian channel on 12h
# - Volume confirmation: 1d volume > 1.5x 20-period median volume (avoid low-participation breakouts)
# - Regime filter: 1d Choppiness Index > 61.8 (range) for mean reversion at channel extremes, < 38.2 (trend) for breakout continuation
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Donchian breakouts capture trends, chop regime adds mean reversion in ranges, volume confirms validity

name = "12h_1d_donchian_volume_chop_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA50 for trend context
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume spike: volume > 1.5x 20-period median
    median_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    volume_spike = volume_1d > (1.5 * median_volume_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # 1d Choppiness Index (CHOP)
    # True Range = max(high-low, abs(high-previous close), abs(low-previous close))
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # CHOP = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    denominator = atr_14 * 14
    chop = np.where(denominator > 0,
                    100 * np.log10(sum_tr_14 / denominator) / np.log10(14),
                    50)  # neutral when undefined
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 12h Donchian channel (20-period)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    highest_high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian lower band OR chop > 61.8 (range) AND price < EMA50
            if (close_12h[i] < lowest_low_20[i] or
                (chop_aligned[i] > 61.8 and close_12h[i] < ema_50_aligned[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian upper band OR chop > 61.8 (range) AND price > EMA50
            if (close_12h[i] > highest_high_20[i] or
                (chop_aligned[i] > 61.8 and close_12h[i] > ema_50_aligned[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with volume confirmation
            # Long: price breaks above upper band AND volume spike
            if (close_12h[i] > highest_high_20[i] and
                volume_spike_aligned[i]):
                # In strong trend (chop < 38.2) or mean reversion setup (chop > 61.8 AND price < EMA50)
                if chop_aligned[i] < 38.2 or (chop_aligned[i] > 61.8 and close_12h[i] < ema_50_aligned[i]):
                    position = 1
                    signals[i] = 0.25
            # Short: price breaks below lower band AND volume spike
            elif (close_12h[i] < lowest_low_20[i] and
                  volume_spike_aligned[i]):
                # In strong trend (chop < 38.2) or mean reversion setup (chop > 61.8 AND price > EMA50)
                if chop_aligned[i] < 38.2 or (chop_aligned[i] > 61.8 and close_12h[i] > ema_50_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals