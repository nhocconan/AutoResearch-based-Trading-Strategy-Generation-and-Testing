#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and 12h ATR filter
# - Entry: Donchian(20) breakout on 4h (long on upper band, short on lower band)
# - Confirmation: 12h volume > 1.3x 20-period average AND 12h ATR > 0.008 (volatility filter)
# - Exit: Opposite Donchian touch OR ATR-based stoploss (1.5x ATR)
# - Position size: 0.25 (discrete level to minimize fee churn)
# - Target: 25-35 trades/year on 4h (100-140 total) to avoid overtrading
# - Works in bull (breakouts continue) and bear (volatility spikes confirm breakdowns)

name = "4h_12h_donchian_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h indicators
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h ATR(14) for volatility filter
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h Donchian(20) channels for alignment reference
    highest_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # 12h Volume > 1.3x 20-period average
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_12h > (1.3 * avg_volume_20)
    
    # 12h ATR filter: only trade when volatility is sufficient (> 0.8% of price)
    atr_threshold = 0.008 * close_12h
    vol_filter = atr_12h > atr_threshold
    
    # Align all 12h indicators to 4h
    highest_20_aligned = align_htf_to_ltf(prices, df_12h, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_12h, lowest_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike.astype(float))
    vol_filter_aligned = align_htf_to_ltf(prices, df_12h, vol_filter.astype(float))
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(vol_filter_aligned[i]) or
            np.isnan(atr_12h_aligned[i]) or atr_12h_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: opposite Donchian touch or ATR stoploss
            if low[i] <= lowest_20_aligned[i]:  # Touch opposite band
                position = 0
                signals[i] = 0.0
            elif high[i] >= entry_price + (1.5 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: opposite Donchian touch or ATR stoploss
            if high[i] >= highest_20_aligned[i]:  # Touch opposite band
                position = 0
                signals[i] = 0.0
            elif low[i] <= entry_price - (1.5 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and volatility filter
            if (high[i] >= highest_20_aligned[i] and  # Break above upper band
                volume_spike_aligned[i] and         # Volume confirmation
                vol_filter_aligned[i]):             # Sufficient volatility
                position = 1
                entry_price = high[i]
                atr_stop = atr_12h_aligned[i]
                signals[i] = 0.25
            elif (low[i] <= lowest_20_aligned[i] and   # Break below lower band
                  volume_spike_aligned[i] and         # Volume confirmation
                  vol_filter_aligned[i]):             # Sufficient volatility
                position = -1
                entry_price = low[i]
                atr_stop = atr_12h_aligned[i]
                signals[i] = -0.25
    
    return signals