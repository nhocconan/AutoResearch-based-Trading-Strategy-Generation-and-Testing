#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and chop regime filter
# - Uses 1d Donchian channels for breakout signals (long above 20-period high, short below 20-period low)
# - Confirms with 1w volume > 2.0x 10-period average (strong institutional participation)
# - Filters by 1d choppiness index: trade only when CHOP > 61.8 (range) OR CHOP < 38.2 (trend)
# - Exits when price touches opposite Donchian level or ATR-based stoploss (2.0x ATR)
# - Position size: 0.25 (25% of capital) for controlled risk
# - Target: 15-30 trades/year on 1d timeframe (60-120 total over 4 years) to minimize fee drag
# - Works in bull markets (breakouts continue) and bear markets (breakdowns continue)
# - Donchian channels provide robust structure that adapts to volatility regimes

name = "1d_1w_donchian_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    volume_1d = prices['volume'].values
    
    # 1d True Range for ATR and chop
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # 1d ATR(14) for stoploss
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d Donchian channels (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # 1d Volume > 1.5x 20-period average (stricter for fewer trades)
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * avg_volume_20)
    
    # 1d Choppiness Index(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = np.where((highest_14 - lowest_14) > 0, highest_14 - lowest_14, 1e-10)
    chop = 100 * np.log10(sum_tr_14 / chop_denom) / np.log10(14)
    chop_range = chop > 61.8  # range-bound market
    chop_trend = chop < 38.2  # trending market
    
    # Pre-compute 1w indicators
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # 1w Volume > 2.0x 10-period average (strong confirmation)
    avg_volume_10_1w = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    volume_spike_1w = volume_1w > (2.0 * avg_volume_10_1w)
    
    # Align all indicators to 1d
    donchian_high_aligned = align_htf_to_ltf(prices, prices, donchian_high)  # 1d to 1d = identity
    donchian_low_aligned = align_htf_to_ltf(prices, prices, donchian_low)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, prices, volume_spike_1d.astype(float))
    chop_range_aligned = align_htf_to_ltf(prices, prices, chop_range.astype(float))
    chop_trend_aligned = align_htf_to_ltf(prices, prices, chop_trend.astype(float))
    atr_1d_aligned = align_htf_to_ltf(prices, prices, atr_1d)
    volume_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(chop_range_aligned[i]) or
            np.isnan(chop_trend_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(volume_spike_1w_aligned[i]) or atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: opposite Donchian touch (low) or ATR stoploss
            if low_1d[i] <= donchian_low_aligned[i]:  # Touch opposite band
                position = 0
                signals[i] = 0.0
            elif high_1d[i] >= entry_price + (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: opposite Donchian touch (high) or ATR stoploss
            if high_1d[i] >= donchian_high_aligned[i]:  # Touch opposite band
                position = 0
                signals[i] = 0.0
            elif low_1d[i] <= entry_price - (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation (1d and 1w) and regime filter
            if (high_1d[i] >= donchian_high_aligned[i] and  # Break above upper band
                volume_spike_1d_aligned[i] and         # 1d volume confirmation
                volume_spike_1w_aligned[i] and         # 1w volume confirmation
                (chop_range_aligned[i] or chop_trend_aligned[i])):  # Either regime
                position = 1
                entry_price = high_1d[i]
                atr_stop = atr_1d_aligned[i]
                signals[i] = 0.25
            elif (low_1d[i] <= donchian_low_aligned[i] and   # Break below lower band
                  volume_spike_1d_aligned[i] and         # 1d volume confirmation
                  volume_spike_1w_aligned[i] and         # 1w volume confirmation
                  (chop_range_aligned[i] or chop_trend_aligned[i])):  # Either regime
                position = -1
                entry_price = low_1d[i]
                atr_stop = atr_1d_aligned[i]
                signals[i] = -0.25
    
    return signals