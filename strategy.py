#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and chop regime filter
# - Uses 1d Donchian channels for breakout signals (long above 20-day high, short below 20-day low)
# - Confirms with 1w volume > 1.5x 20-period average (institutional participation on weekly timeframe)
# - Filters by 1d choppiness index: trade only when CHOP > 61.8 (range) or CHOP < 38.2 (trend)
# - Exits when price touches opposite Donchian level or ATR-based stoploss (2x ATR)
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 20-50 trades/year on 1d timeframe (80-200 total over 4 years) to minimize fee drag
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
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators (primary timeframe)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d True Range for ATR and chop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # 1d ATR(14) for stoploss
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d Volume > 1.5x 20-period average
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume > (1.5 * avg_volume_20)
    
    # 1d Choppiness Index(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = np.where((highest_14 - lowest_14) > 0, highest_14 - lowest_14, 1e-10)
    chop = 100 * np.log10(sum_tr_14 / chop_denom) / np.log10(14)
    chop_range = chop > 61.8  # range-bound market
    chop_trend = chop < 38.2  # trending market
    
    # Pre-compute 1w indicators (HTF)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # 1w Volume > 1.5x 20-period average
    avg_volume_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_spike_1w = volume_1w > (1.5 * avg_volume_20_1w)
    
    # Align 1w volume spike to 1d
    volume_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_spike_1d[i]) or np.isnan(chop_range[i]) or
            np.isnan(chop_trend[i]) or np.isnan(atr_1d[i]) or
            np.isnan(volume_spike_1w_aligned[i]) or
            atr_1d[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: opposite Donchian touch (low) or ATR stoploss
            if low[i] <= donchian_low[i]:  # Touch opposite band
                position = 0
                signals[i] = 0.0
            elif high[i] >= entry_price + (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: opposite Donchian touch (high) or ATR stoploss
            if high[i] >= donchian_high[i]:  # Touch opposite band
                position = 0
                signals[i] = 0.0
            elif low[i] <= entry_price - (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation (both 1d and 1w) and regime filter
            if (high[i] >= donchian_high[i] and  # Break above upper band
                volume_spike_1d[i] and         # 1d volume confirmation
                volume_spike_1w_aligned[i] and # 1w volume confirmation
                (chop_range[i] or chop_trend[i])):  # Either regime
                position = 1
                entry_price = high[i]
                atr_stop = atr_1d[i]
                signals[i] = 0.25
            elif (low[i] <= donchian_low[i] and   # Break below lower band
                  volume_spike_1d[i] and         # 1d volume confirmation
                  volume_spike_1w_aligned[i] and # 1w volume confirmation
                  (chop_range[i] or chop_trend[i])):  # Either regime
                position = -1
                entry_price = low[i]
                atr_stop = atr_1d[i]
                signals[i] = -0.25
    
    return signals