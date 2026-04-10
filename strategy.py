#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w pivot direction filter and volume confirmation
# - Primary signal: Price breaks above/below 20-period Donchian channel on 6h
# - HTF filter: 1w Camarilla pivot direction (price > weekly pivot = bullish bias, < weekly pivot = bearish bias)
# - Volume confirmation: 6h volume > 1.8x 20-period average volume (avoid low-participation breakouts)
# - Works in bull/bear: Pivot direction adapts to weekly trend, volume confirms participation
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines

name = "6h_1w_camarilla_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w Camarilla pivots (based on prior week OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Camarilla pivot levels for weekly timeframe
    # Pivot = (High + Low + Close) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Range = High - Low
    range_1w = high_1w - low_1w
    # Resistance levels: R3 = Close + Range * 1.1/4, R4 = Close + Range * 1.1/2
    # Support levels: S3 = Close - Range * 1.1/4, S4 = Close - Range * 1.1/2
    r3_1w = close_1w + range_1w * 1.1 / 4
    r4_1w = close_1w + range_1w * 1.1 / 2
    s3_1w = close_1w - range_1w * 1.1 / 4
    s4_1w = close_1w - range_1w * 1.1 / 2
    
    # Bullish bias: price above weekly pivot AND above S3 (bullish zone)
    # Bearish bias: price below weekly pivot AND below R3 (bearish zone)
    bullish_bias = (close_1w > pivot_1w) & (close_1w > s3_1w)
    bearish_bias = (close_1w < pivot_1w) & (close_1w < r3_1w)
    
    # Align HTF bias to 6h timeframe (wait for weekly bar to close)
    bullish_bias_aligned = align_htf_to_ltf(prices, df_1w, bullish_bias.astype(float))
    bearish_bias_aligned = align_htf_to_ltf(prices, df_1w, bearish_bias.astype(float))
    
    # Pre-compute 6h Donchian Channel (20)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute 6h volume spike filter
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_6h > (1.8 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(bullish_bias_aligned[i]) or np.isnan(bearish_bias_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to Donchian mean OR weekly bias turns bearish
            if close_6h[i] < donchian_mid[i] or bearish_bias_aligned[i] == 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to Donchian mean OR weekly bias turns bullish
            if close_6h[i] > donchian_mid[i] or bullish_bias_aligned[i] == 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with volume spike and weekly bias alignment
            if volume_spike[i]:
                # Long: price breaks above upper Donchian band with bullish weekly bias
                if close_6h[i] > donchian_high[i] and bullish_bias_aligned[i] == 1.0:
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: price breaks below lower Donchian band with bearish weekly bias
                elif close_6h[i] < donchian_low[i] and bearish_bias_aligned[i] == 1.0:
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals