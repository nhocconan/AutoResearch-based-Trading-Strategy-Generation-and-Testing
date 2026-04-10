#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and choppiness regime filter
# - Long when price breaks above Donchian(20) high AND 1d volume > 2.0x 20-period average AND chop(14) > 61.8 (range regime)
# - Short when price breaks below Donchian(20) low AND 1d volume > 2.0x 20-period average AND chop(14) > 61.8 (range regime)
# - Exit when price crosses Donchian(20) midpoint OR chop(14) < 38.2 (trending regime)
# - Uses discrete position sizing (0.25) to control drawdown and minimize fee churn
# - Targets ~12-25 trades/year (50-100 total over 4 years) to stay within HARD MAX: 200 total
# - Donchian breakouts capture momentum; volume confirms institutional participation
# - Chop filter ensures we only trade in ranging markets where mean reversion works
# - Works in both bull and bear markets by trading range-bound conditions

name = "12h_1d_donchian_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period) on 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian high: highest high over last 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low over last 20 periods
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint: average of high and low
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Pre-compute 1d volume spike: > 2.0x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * volume_20_avg_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Pre-compute Choppiness Index (14-period) on 1d data
    # Chop = 100 * log10(sum(ATR(14)) / (log10(highest_high - lowest_low) * 14))
    # Simplified: Chop = 100 * log10(ATR_sum / (log10(range) * period))
    # We'll use a practical approximation: Chop = 100 * log10(sum(true_range) / (log10(hh_ll) * 14))
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with close_1d
    
    # ATR(14) = sum of true range over 14 periods
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range14 = hh14 - ll14
    
    # Choppiness Index: 100 * log10(atr14 / (log10(range14) * 14)) / log10(100)
    # Avoid division by zero and log of zero/negative
    log_range = np.log10(np.maximum(range14, 1e-10))
    chop = 100 * (np.log10(np.maximum(atr14, 1e-10)) / (log_range * 14))
    chop = np.where((range14 > 0) & (atr14 > 0), chop, 50.0)  # default to 50 when invalid
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high AND volume spike AND chop > 61.8 (range)
            if (close[i] > donchian_high[i] and 
                volume_spike_1d_aligned[i] and 
                chop_aligned[i] > 61.8):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low AND volume spike AND chop > 61.8 (range)
            elif (close[i] < donchian_low[i] and 
                  volume_spike_1d_aligned[i] and 
                  chop_aligned[i] > 61.8):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price crosses Donchian midpoint (mean reversion complete)
            # 2. Chop < 38.2 (trending regime - exit range play)
            if position == 1:
                if close[i] < donchian_mid[i] or chop_aligned[i] < 38.2:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:
                if close[i] > donchian_mid[i] or chop_aligned[i] < 38.2:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals