#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
# - Long when price breaks above Camarilla H3 (1d) AND 1d volume > 1.5x 20-bar avg AND chop < 61.8 (trending)
# - Short when price breaks below Camarilla L3 (1d) AND 1d volume > 1.5x 20-bar avg AND chop < 61.8 (trending)
# - Exit when price returns to Camarilla pivot point (1d) or chop > 61.8 (range)
# - Uses discrete position sizing (0.25) to balance return and drawdown
# - Camarilla levels from 1d provide institutional support/resistance
# - Volume confirmation avoids low-liquidity false breakouts
# - Chop filter ensures we only trade in trending markets, avoiding whipsaws in ranges
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Works in both bull and bear markets: breakouts work in trends, chop filter avoids range-bound losses

name = "4h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # Pivot = (high + low + close) / 3
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * range_1d
    camarilla_l3 = close_1d - 1.1 * range_1d
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * volume_20_avg_1d)
    
    # Pre-compute chop regime filter on 1d: chop < 61.8 = trending (trade breakouts)
    # Chop = 100 * log10(sum(ATR,14) / (max(high,14) - min(low,14))) / log10(14)
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.absolute(high_1d - np.roll(close_1d, 1)),
                       np.absolute(low_1d - np.roll(close_1d, 1)))
    tr_1d[0] = high_1d[0] - low_1d[0]  # First bar
    
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = max_high_14 - min_low_14
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)  # Avoid div by zero
    chop_1d = 100 * np.log10(atr_14_1d / chop_denominator) / np.log10(14)
    chop_1d = np.where(chop_denominator == 0, 50, chop_1d)  # Handle zero range
    chop_trending = chop_1d < 61.8  # Trending market
    
    # Align HTF indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    chop_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_trending)
    
    # Get 4h price data
    open_prices = prices['open'].values
    high_prices = prices['high'].values
    low_prices = prices['low'].values
    close_prices = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or
            np.isnan(chop_trending_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND volume spike AND trending market
            if (close_prices[i] > camarilla_h3_aligned[i] and 
                vol_spike_1d_aligned[i] and 
                chop_trending_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L3 AND volume spike AND trending market
            elif (close_prices[i] < camarilla_l3_aligned[i] and 
                  vol_spike_1d_aligned[i] and 
                  chop_trending_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit conditions
            # Exit when price returns to pivot OR market becomes ranging (chop > 61.8)
            exit_signal = (abs(close_prices[i] - camarilla_pivot_aligned[i]) < 0.001 * close_prices[i]) or \
                          (~chop_trending_aligned[i])
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals