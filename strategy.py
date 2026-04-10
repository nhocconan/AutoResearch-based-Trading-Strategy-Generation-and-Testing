#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume confirmation + 1d chop regime filter
# - Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-period average AND 1d chop < 38.2 (trending)
# - Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-period average AND 1d chop < 38.2 (trending)
# - Exit when price crosses Donchian(20) midpoint with volume confirmation
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Donchian breakouts capture strong trends in both bull and bear markets
# - Volume confirmation ensures breakouts have conviction
# - Chop filter ensures we only trade in trending conditions where breakouts work

name = "12h_1d_donchian_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute 1d volume confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (1.5 * vol_ma_1d)
    
    # Pre-compute 1d chop regime (choppiness index)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - np.roll(close_1d, 1)[1:])
    tr3 = np.abs(low_1d[1:] - np.roll(close_1d, 1)[1:])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # first element is NaN
    
    # ATR(14)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # Max(high) - Min(low) over 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_max_min = max_high - min_low
    
    # Chop = 100 * log10(tr_sum / range_max_min) / log10(14)
    chop = 100 * np.log10(tr_sum / range_max_min) / np.log10(14)
    chop = np.concatenate([np.full(13, np.nan), chop[13:]])  # align indices
    
    # Chop regime: < 38.2 = trending (good for breakout strategies)
    chop_trend = chop < 38.2
    
    # Align HTF indicators to 12h timeframe
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    chop_trend_aligned = align_htf_to_ltf(prices, df_1d, chop_trend)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(chop_trend_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Donchian high AND volume spike AND chop trend
            if (close[i] > donchian_high[i] and 
                volume_spike_1d_aligned[i] and 
                chop_trend_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Donchian low AND volume spike AND chop trend
            elif (close[i] < donchian_low[i] and 
                  volume_spike_1d_aligned[i] and 
                  chop_trend_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price crosses Donchian midpoint with volume confirmation
            exit_long = (position == 1 and 
                        close[i] < donchian_mid[i] and 
                        volume_spike_1d_aligned[i])
            exit_short = (position == -1 and 
                         close[i] > donchian_mid[i] and 
                         volume_spike_1d_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals