#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1w pivot direction + volume confirmation
# - Primary signal: Price breaks above/below 20-period Donchian channel on 6h
# - Direction filter: Only take breakouts aligned with 1w Camarilla pivot bias
#   (bullish if price > weekly pivot, bearish if price < weekly pivot)
# - Volume confirmation: 1d volume > 1.3x 20-period average volume
# - Works in bull/bear: Weekly pivot adapts to long-term trend; volume ensures participation
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines

name = "6h_1w_1d_donchian_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w Camarilla pivot levels (using prior week OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Weekly pivot = (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3
    # Weekly range = H - L
    weekly_range = high_1w - low_1w
    
    # Camarilla levels (based on prior week)
    camarilla_h4 = weekly_pivot + (weekly_range * 1.1 / 2)  # R4
    camarilla_l4 = weekly_pivot - (weekly_range * 1.1 / 2)  # S4
    camarilla_h3 = weekly_pivot + (weekly_range * 1.1 / 4)  # R3
    camarilla_l3 = weekly_pivot - (weekly_range * 1.1 / 4)  # S3
    
    # Pivot bias: 1 = bullish (price > weekly pivot), -1 = bearish (price < weekly pivot)
    pivot_bias = np.where(close_1w > weekly_pivot, 1, -1)
    
    # Align HTF data to 6h timeframe
    pivot_bias_aligned = align_htf_to_ltf(prices, df_1w, pivot_bias)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.3 * avg_volume_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Pre-compute 6h Donchian Channel (20)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute 6h ATR(20) for stoploss
    tr_6h1 = high_6h - low_6h
    tr_6h2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr_6h3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr_6h = np.maximum(tr_6h1, np.maximum(tr_6h2, tr_6h3))
    tr_6h[0] = tr_6h1[0]
    atr_20 = pd.Series(tr_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(pivot_bias_aligned[i]) or np.isnan(volume_spike_aligned[i]) or
            np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Donchian mean reversion OR stoploss hit
            if close_6h[i] < donchian_mid[i] or close_6h[i] < entry_price - 2.5 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Donchian mean reversion OR stoploss hit
            if close_6h[i] > donchian_mid[i] or close_6h[i] > entry_price + 2.5 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with volume spike and pivot bias
            if volume_spike_aligned[i]:
                # Long breakout: price above upper Donchian band with bullish weekly bias
                if (close_6h[i] > donchian_high[i] and 
                    pivot_bias_aligned[i] == 1):
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short breakout: price below lower Donchian band with bearish weekly bias
                elif (close_6h[i] < donchian_low[i] and 
                      pivot_bias_aligned[i] == -1):
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals