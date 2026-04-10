#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and chop regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 1.8x 20-period average AND chop > 61.8 (ranging market)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 1.8x 20-period average AND chop > 61.8
# - Exit when price reverses to Camarilla H4/L4 levels or chop < 38.2 (trending market)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Camarilla levels from daily timeframe provide institutional support/resistance
# - Volume confirmation reduces false breakouts in ranging markets
# - Chop filter ensures we trade in ranging conditions where mean reversion works
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_1d_camarilla_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H4 = Close + 1.5 * (High - Low)
    # H3 = Close + 1.25 * (High - Low)
    # H2 = Close + 1.166 * (High - Low)
    # H1 = Close + 1.0833 * (High - Low)
    # L1 = Close - 1.0833 * (High - Low)
    # L2 = Close - 1.166 * (High - Low)
    # L3 = Close - 1.25 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels using previous day's values (shifted by 1)
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d, 1)
    
    # First day has no previous data
    high_prev[0] = high_1d[0]
    low_prev[0] = low_1d[0]
    close_prev[0] = close_1d[0]
    
    rang = high_prev - low_prev
    
    # Camarilla levels
    H4 = close_prev + 1.5 * rang
    H3 = close_prev + 1.25 * rang
    L3 = close_prev - 1.25 * rang
    L4 = close_prev - 1.5 * rang
    H4_alt = close_prev + 1.0833 * rang  # H1
    L4_alt = close_prev - 1.0833 * rang  # L1
    
    # Pre-compute 1d volume confirmation (20-period average)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (1.8 * vol_ma_1d)
    
    # Pre-compute 1d Chop Index (choppiness) - 14 period
    # Chop = 100 * log10(sum(ATR14) / (max(high14) - min(low14))) / log10(14)
    def true_range(high_arr, low_arr, close_arr):
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr1[0] = 0
        tr2[0] = 0
        tr3[0] = 0
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr_1d = true_range(high_1d, low_1d, close_1d)
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = max_high_14 - min_low_14
    range_14[range_14 == 0] = 1e-10
    
    chop_1d = 100 * np.log10(pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values / range_14) / np.log10(14)
    
    # Chop regime: > 61.8 = ranging (good for mean reversion), < 38.2 = trending
    chop_regime_ranging = chop_1d > 61.8
    chop_regime_trending = chop_1d < 38.2
    
    # Align HTF indicators to 4h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    chop_regime_ranging_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_ranging)
    chop_regime_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_trending)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(chop_regime_ranging_aligned[i]) or
            np.isnan(chop_regime_trending_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND volume spike AND chop > 61.8 (ranging)
            if (close[i] > H3_aligned[i] and 
                volume_spike_1d_aligned[i] and 
                chop_regime_ranging_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below L3 AND volume spike AND chop > 61.8 (ranging)
            elif (close[i] < L3_aligned[i] and 
                  volume_spike_1d_aligned[i] and 
                  chop_regime_ranging_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: 
            # 1. Price reaches opposite Camarilla level (H4 for long, L4 for short)
            # 2. Chop regime shifts to trending (< 38.2)
            exit_long = (position == 1 and (close[i] >= H4_aligned[i] or chop_regime_trending_aligned[i]))
            exit_short = (position == -1 and (close[i] <= L4_aligned[i] or chop_regime_trending_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals