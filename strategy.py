#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d regime filter
# - Uses 4h Camarilla pivot levels (H3/L3) for breakout signals
# - Confirms with 1d volume > 2.0x 24-period average (strong participation)
# - Filters by 1d ADX > 25 to ensure trending market (avoid chop)
# - Exits on opposite Camarilla level touch (H3/L3) or time-based exit (24 bars)
# - Position size: 0.20 (20% of capital) to minimize drawdown
# - Target: 15-35 trades/year on 1h timeframe (60-140 total over 4 years)
# - Works in bull markets (breakouts continue) and bear markets (breakdowns continue)
# - Camarilla pivots provide mathematically derived support/resistance levels

name = "1h_4h_1d_camarilla_volume_adx_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h indicators
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h Camarilla pivots (based on previous day's OHLC)
    # H4 = close + 1.1*(high-low)*1.1/2, L4 = close - 1.1*(high-low)*1.1/2
    # H3 = close + 1.1*(high-low)*1.1/4, L3 = close - 1.1*(high-low)*1.1/4
    # H2 = close + 1.1*(high-low)*1.1/6, L2 = close - 1.1*(high-low)*1.1/6
    # H1 = close + 1.1*(high-low)*1.1/12, L1 = close - 1.1*(high-low)*1.1/12
    # We'll use H3/L3 for breakouts
    prev_close = np.roll(close_4h, 1)
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close[0] = close_4h[0]
    prev_high[0] = high_4h[0]
    prev_low[0] = low_4h[0]
    
    camarilla_range = prev_high - prev_low
    camarilla_H3 = prev_close + 1.1 * camarilla_range * 1.1 / 4
    camarilla_L3 = prev_close - 1.1 * camarilla_range * 1.1 / 4
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Volume > 2.0x 24-period average
    avg_volume_24 = pd.Series(volume_1d).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume_1d > (2.0 * avg_volume_24)
    
    # 1d ADX(14) for trend strength
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM-
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF indicators to 1h
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_L3)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    bars_in_trade = 0
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(adx_aligned[i]) or
            adx_aligned[i] < 25):  # Require trending market
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: opposite Camarilla touch (L3) or time-based exit (24 bars)
            if low[i] <= camarilla_L3_aligned[i]:  # Touch opposite band
                position = 0
                bars_in_trade = 0
                signals[i] = 0.0
            elif bars_in_trade >= 24:  # Time-based exit
                position = 0
                bars_in_trade = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                bars_in_trade += 1
                
        elif position == -1:  # Short position
            # Exit conditions: opposite Camarilla touch (H3) or time-based exit (24 bars)
            if high[i] >= camarilla_H3_aligned[i]:  # Touch opposite band
                position = 0
                bars_in_trade = 0
                signals[i] = 0.0
            elif bars_in_trade >= 24:  # Time-based exit
                position = 0
                bars_in_trade = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
                bars_in_trade += 1
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation and trend filter
            if (high[i] >= camarilla_H3_aligned[i] and  # Break above H3
                volume_spike_aligned[i]):             # Volume confirmation
                position = 1
                entry_price = high[i]
                bars_in_trade = 1
                signals[i] = 0.20
            elif (low[i] <= camarilla_L3_aligned[i] and   # Break below L3
                  volume_spike_aligned[i]):             # Volume confirmation
                position = -1
                entry_price = low[i]
                bars_in_trade = 1
                signals[i] = -0.20
    
    return signals