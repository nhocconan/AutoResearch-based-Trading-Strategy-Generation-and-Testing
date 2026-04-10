#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout direction + 1h volume spike + session filter
# - Primary signal direction: Price breaks above/below 20-period Donchian channel on 4h (trend filter)
# - Entry timing: 1h volume > 2.0x 20-period average volume (avoid low-participation breakouts)
# - Session filter: Only trade 08-20 UTC (avoid low-volume Asian session noise)
# - Position size: 0.20 discrete level to minimize fee churn
# - Target: 15-30 trades/year (60-120 total over 4 years) per 1h strategy guidelines
# - Stoploss: exit when price moves against position by 2.0x ATR(20) on 1h

name = "1h_4h_donchian_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h Donchian Channel (20)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute 4h ATR(20) for trend strength (optional filter)
    tr_4h1 = high_4h - low_4h
    tr_4h2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr_4h3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    tr_4h[0] = tr_4h1[0]
    atr_20_4h = pd.Series(tr_4h).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 1h volume spike filter
    volume_1h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1h > (2.0 * avg_volume_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Align HTF indicators to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or np.isnan(volume_spike) or
            np.isnan(session_filter[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: skip outside 08-20 UTC
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Donchian mean reversion OR stoploss hit (using 1h ATR approximation)
            if prices['close'].iloc[i] < donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: Donchian mean reversion OR stoploss hit
            if prices['close'].iloc[i] > donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Donchian breakouts with volume spike
            # Long: price breaks above upper Donchian band
            if prices['close'].iloc[i] > donchian_high_aligned[i] and volume_spike[i]:
                position = 1
                entry_price = prices['close'].iloc[i]
                signals[i] = 0.20
            # Short: price breaks below lower Donchian band
            elif prices['close'].iloc[i] < donchian_low_aligned[i] and volume_spike[i]:
                position = -1
                entry_price = prices['close'].iloc[i]
                signals[i] = -0.20
    
    return signals