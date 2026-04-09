#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
# - Uses 1d Camarilla pivot levels (H3/L3) for entry (long above H3, short below L3)
# - Confirms with 1d volume > 1.8x 20-period average (strong institutional participation)
# - Filters by 1d choppiness index: only trade when CHOP < 40 (strong trending) or CHOP > 65 (strong ranging)
# - Exits when price touches opposite Camarilla level (H3/L3) or ATR-based stoploss (1.5x ATR)
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 12-30 trades/year on 12h timeframe (50-120 total over 4 years) to minimize fee drag
# - Works in bull markets (breakouts continue) and bear markets (breakdowns continue)

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute HTF indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d True Range for ATR and Chop
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # 1d ATR(14)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d Camarilla levels (based on previous day)
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # 1d Volume > 1.8x 20-period average
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.8 * avg_volume_20)
    
    # 1d Choppiness Index(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = np.where((highest_14 - lowest_14) > 0, highest_14 - lowest_14, 1e-10)
    chop = 100 * np.log10(sum_tr_14 / chop_denom) / np.log10(14)
    chop_strong_trend = chop < 40   # Strong trending market
    chop_strong_range = chop > 65   # Strong ranging market
    
    # Align all 1d indicators to 12h (2 bars per day)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    chop_strong_trend_aligned = align_htf_to_ltf(prices, df_1d, chop_strong_trend.astype(float))
    chop_strong_range_aligned = align_htf_to_ltf(prices, df_1d, chop_strong_range.astype(float))
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_strong_trend_aligned[i]) or
            np.isnan(chop_strong_range_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: touch L3 or ATR stoploss
            if low[i] <= camarilla_l3_aligned[i]:  # Touch opposite band
                position = 0
                signals[i] = 0.0
            elif high[i] >= entry_price + (1.5 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: touch H3 or ATR stoploss
            if high[i] >= camarilla_h3_aligned[i]:  # Touch opposite band
                position = 0
                signals[i] = 0.0
            elif low[i] <= entry_price - (1.5 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation and regime filter
            if (high[i] >= camarilla_h3_aligned[i] and  # Break above H3
                volume_spike_aligned[i] and         # Volume confirmation
                (chop_strong_trend_aligned[i] or chop_strong_range_aligned[i])):  # Either regime
                position = 1
                entry_price = high[i]
                atr_stop = atr_1d_aligned[i]
                signals[i] = 0.25
            elif (low[i] <= camarilla_l3_aligned[i] and   # Break below L3
                  volume_spike_aligned[i] and         # Volume confirmation
                  (chop_strong_trend_aligned[i] or chop_strong_range_aligned[i])):  # Either regime
                position = -1
                entry_price = low[i]
                atr_stop = atr_1d_aligned[i]
                signals[i] = -0.25
    
    return signals