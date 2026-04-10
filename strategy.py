#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and 1d chop regime filter
# - Long when price breaks above Camarilla H3 level AND volume > 1.8x 20-period average AND 1d chop > 61.8 (range)
# - Short when price breaks below Camarilla L3 level AND volume > 1.8x 20-period average AND 1d chop > 61.8 (range)
# - Exit when price returns to Camarilla H4/L4 levels or opposite signal with volume confirmation
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Camarilla pivots derived from 1d OHLC provide institutional support/resistance levels
# - Volume confirmation reduces false breakouts
# - Chop filter ensures trades occur only in ranging markets where mean reversion at extremes is effective
# - Works in both bull and bear markets by capturing mean reversion at key levels

name = "4h_1d_camarilla_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Pre-compute 4h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Pre-compute 1d Camarilla pivot levels from previous day's OHLC
    # Camarilla levels: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    #                L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_h4 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 2
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 4
    camarilla_l4 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 2
    
    # Align HTF Camarilla levels to 4h timeframe (completed 1d bar only)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 1d chop regime (choppiness index)
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
    
    # Align HTF chop regime to 4h timeframe
    chop_range_aligned = align_htf_to_ltf(prices, df_1d, chop_range)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop_range_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Camarilla H3 AND volume spike AND chop range
            if (close[i] > camarilla_h3_aligned[i-1] and  # breakout above previous period's H3
                volume_spike[i] and 
                chop_range_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Camarilla L3 AND volume spike AND chop range
            elif (close[i] < camarilla_l3_aligned[i-1] and  # breakout below previous period's L3
                  volume_spike[i] and 
                  chop_range_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price returns to Camarilla H4/L4 levels or opposite signal with volume
            exit_long = (position == 1 and 
                        (close[i] <= camarilla_h4_aligned[i] or
                         (close[i] < camarilla_l3_aligned[i] and volume_spike[i])))
            exit_short = (position == -1 and 
                         (close[i] >= camarilla_l4_aligned[i] or
                          (close[i] > camarilla_h3_aligned[i] and volume_spike[i])))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals