#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX + volume spike + 1d chop regime filter
# - Long when TRIX crosses above zero AND volume > 1.5x 20-period average AND 1d chop < 38.2 (trending)
# - Short when TRIX crosses below zero AND volume > 1.5x 20-period average AND 1d chop < 38.2 (trending)
# - Exit when TRIX crosses zero in opposite direction with volume confirmation
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - TRIX identifies momentum changes with reduced whipsaw
# - Volume confirmation ensures breakouts have conviction
# - Chop filter ensures we only trade in trending conditions where momentum works

name = "4h_1d_trix_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h TRIX (15-period triple EMA)
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA1, EMA2, EMA3 for TRIX
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    
    # TRIX = 100 * (EMA3 - previous EMA3) / previous EMA3
    trix = np.zeros_like(close)
    trix[1:] = 100 * (ema3[1:] - ema3[:-1]) / ema3[:-1]
    trix[0] = 0
    # Handle division by zero
    trix = np.where(ema3 == 0, 0, trix)
    
    # Pre-compute 4h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
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
    
    # Chop regime: < 38.2 = trending (good for momentum)
    chop_trending = chop < 38.2
    
    # Align HTF indicators to 4h timeframe
    chop_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_trending)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(trix[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(chop_trending_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: TRIX crosses above zero AND volume spike AND chop trending
            if (trix[i] > 0 and trix[i-1] <= 0 and 
                volume_spike[i] and 
                chop_trending_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: TRIX crosses below zero AND volume spike AND chop trending
            elif (trix[i] < 0 and trix[i-1] >= 0 and 
                  volume_spike[i] and 
                  chop_trending_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when TRIX crosses zero in opposite direction with volume confirmation
            exit_long = (position == 1 and 
                        trix[i] < 0 and trix[i-1] >= 0 and 
                        volume_spike[i])
            exit_short = (position == -1 and 
                         trix[i] > 0 and trix[i-1] <= 0 and 
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