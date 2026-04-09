#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume spike
# - Uses 4h Donchian channels for breakout signals (long above 20-period high, short below 20-period low)
# - Filters by 1d ATR ratio: ATR(7)/ATR(30) > 1.5 to ensure sufficient volatility for breakouts
# - Requires 1d volume > 2.0x 20-period average for institutional confirmation
# - Uses opposite Donchian touch for exit (mean reversion within the channel)
# - Position size: 0.25 (25% of capital) to balance risk and minimize fee drag
# - Target: 15-30 trades/year on 4h timeframe (60-120 total over 4 years) to avoid overtrading
# - Works in bull markets (breakouts continue) and bear markets (breakdowns continue)
# - Volatility filter prevents trading in choppy low-volume environments

name = "4h_1d_donchian_vol_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Pre-compute HTF indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d True Range for ATR calculations
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # 1d ATR(7) and ATR(30) for volatility filter
    atr_7 = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
    atr_30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    
    # 1d ATR ratio: ATR(7)/ATR(30) > 1.5 indicates expanding volatility
    atr_ratio = np.where(atr_30 > 0, atr_7 / atr_30, 0)
    volatility_expanding = atr_ratio > 1.5
    
    # 1d Volume > 2.0x 20-period average (strict for fewer trades)
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * avg_volume_20)
    
    # 1d Donchian channels (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align all 1d indicators to 4h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    volatility_expanding_aligned = align_htf_to_ltf(prices, df_1d, volatility_expanding.astype(float))
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(40, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(volatility_expanding_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when price touches lower Donchian band (mean reversion)
            if low[i] <= donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price touches upper Donchian band (mean reversion)
            if high[i] >= donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and volatility filter
            if (high[i] >= donchian_high_aligned[i] and  # Break above upper band
                volume_spike_aligned[i] and         # Volume confirmation
                volatility_expanding_aligned[i]):   # Expanding volatility
                position = 1
                entry_price = high[i]
                signals[i] = 0.25
            elif (low[i] <= donchian_low_aligned[i] and   # Break below lower band
                  volume_spike_aligned[i] and         # Volume confirmation
                  volatility_expanding_aligned[i]):   # Expanding volatility
                position = -1
                entry_price = low[i]
                signals[i] = -0.25
    
    return signals