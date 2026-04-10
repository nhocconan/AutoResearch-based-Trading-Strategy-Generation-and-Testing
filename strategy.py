#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + volume confirmation + 1d chop regime filter
# - Long when Williams %R < -80 (oversold) AND volume > 1.3x 20-period average AND 1d chop > 61.8 (range)
# - Short when Williams %R > -20 (overbought) AND volume > 1.3x 20-period average AND 1d chop > 61.8 (range)
# - Exit when Williams %R crosses -50 (mean reversion midpoint) with volume confirmation
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Williams %R identifies exhaustion points in ranging markets
# - Volume confirmation ensures breakouts have conviction
# - Chop filter ensures we only trade in ranging conditions where mean reversion works

name = "6h_1d_williamsr_volume_chop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 6h Williams %R (14)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Pre-compute 6h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)
    
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
    
    # Chop regime: > 61.8 = ranging (good for mean reversion at extremes)
    chop_range = chop > 61.8
    
    # Align HTF indicators to 6h timeframe
    chop_range_aligned = align_htf_to_ltf(prices, df_1d, chop_range)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(chop_range_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Williams %R oversold AND volume spike AND chop range
            if (williams_r[i] < -80 and 
                volume_spike[i] and 
                chop_range_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R overbought AND volume spike AND chop range
            elif (williams_r[i] > -20 and 
                  volume_spike[i] and 
                  chop_range_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when Williams %R crosses -50 (mean reversion midpoint) with volume confirmation
            exit_long = (position == 1 and 
                        williams_r[i] > -50 and 
                        volume_spike[i])
            exit_short = (position == -1 and 
                         williams_r[i] < -50 and 
                         volume_spike[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals