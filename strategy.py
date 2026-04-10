#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and choppiness regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 1.5x 20-period average AND choppiness index > 61.8 (ranging market)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 1.5x 20-period average AND choppiness index > 61.8
# - Exit when price returns to Camarilla Pivot point (mean reversion to center)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Camarilla pivots work well in ranging markets which dominate 2025 BTC/ETH action
# - Volume confirmation reduces false breakouts
# - Choppiness filter ensures we only trade in ranging regimes where mean reversion works

name = "4h_1d_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Pre-compute 1d OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d timeframe
    # Camarilla equations: 
    # H4 = close + 1.5*(high-low)
    # H3 = close + 1.1*(high-low)
    # H2 = close + 0.55*(high-low)
    # H1 = close + 0.275*(high-low)
    # Pivot = (high + low + close)/3
    # L1 = close - 0.275*(high-low)
    # L2 = close - 0.55*(high-low)
    # L3 = close - 1.1*(high-low)
    # L4 = close - 1.5*(high-low)
    
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d)
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Pre-compute 1d volume confirmation (20-period average)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (1.5 * vol_ma_1d)
    
    # Pre-compute 1d Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    high_low_1d = high_1d - low_1d
    high_close_1d = np.abs(high_1d - np.roll(close_1d, 1))
    low_close_1d = np.abs(low_1d - np.roll(close_1d, 1))
    high_close_1d[0] = 0
    low_close_1d[0] = 0
    tr_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    
    atr_1d = np.zeros_like(tr_1d)
    atr_1d[13] = np.mean(tr_1d[1:14])  # First ATR value
    for i in range(14, len(tr_1d)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index
    chop_1d = np.zeros_like(atr_1d)
    for i in range(34, len(chop_1d)):  # Need 14+20=34 for stable calculation
        if atr_1d[i] > 0 and sum_tr_14[i] > 0:
            chop_1d[i] = 100 * np.log10(sum_tr_14[i] / (atr_1d[i] * 14)) / np.log10(14)
        else:
            chop_1d[i] = 50  # Neutral value when calculation invalid
    
    # Chop > 61.8 indicates ranging market (good for mean reversion)
    chop_regime = chop_1d > 61.8
    
    # Align HTF indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(chop_regime_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND volume spike AND chop regime (ranging)
            if (close[i] > camarilla_h3_aligned[i] and 
                volume_spike_aligned[i] and 
                chop_regime_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below L3 AND volume spike AND chop regime (ranging)
            elif (close[i] < camarilla_l3_aligned[i] and 
                  volume_spike_aligned[i] and 
                  chop_regime_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot (mean reversion)
            # Exit conditions: price returns to pivot level
            exit_long = (position == 1 and close[i] < camarilla_pivot_aligned[i])
            exit_short = (position == -1 and close[i] > camarilla_pivot_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals