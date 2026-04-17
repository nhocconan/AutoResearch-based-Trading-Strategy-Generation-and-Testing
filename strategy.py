#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Choppiness Index regime filter with 1-hour volume spike and 1-day price action
# In trending markets (CHOP < 38.2), trade breakouts with volume confirmation
# In ranging markets (CHOP > 61.8), trade mean reversion at Bollinger Bands
# Volume spike filters low-conviction moves; regime filter adapts to market conditions
# Target: 25-40 trades/year to minimize fee drag while capturing regime-appropriate moves

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h Choppiness Index (14-period) ===
    df_1h = get_htf_data(prices, '1h')
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # True Range
    tr1 = high_1h - low_1h
    tr2 = np.abs(high_1h - np.concatenate([[close_1h[0]], close_1h[:-1]]))
    tr3 = np.abs(low_1h - np.concatenate([[close_1h[0]], close_1h[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of true ranges over period
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over period
    hh = pd.Series(high_1h).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(atr_sum / (hh - ll)) / log10(period)
    # Avoid division by zero
    hl_range = hh - ll
    hl_range_safe = np.where(hl_range == 0, 1e-10, hl_range)
    chop = 100 * np.log10(atr_sum / hl_range_safe) / np.log10(14)
    
    chop_1h_aligned = align_htf_to_ltf(prices, df_1h, chop)
    
    # === 1h Bollinger Bands (20, 2.0) ===
    sma_20 = pd.Series(close_1h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1h).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2.0 * std_20
    bb_lower = sma_20 - 2.0 * std_20
    
    bb_upper_aligned = align_htf_to_ltf(prices, df_1h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1h, bb_lower)
    
    # === 1h Volume Spike (vs 20-period average) ===
    volume_1h = df_1h['volume'].values
    vol_ma_20 = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1h, vol_ma_20)
    
    # === 4h Donchian Channel (20-period) for breakouts ===
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(chop_1h_aligned[i]) or np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(high_20[i]) or np.isnan(low_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 1h volume (avoid calling get_htf_data in loop)
        volume_1h_aligned = align_htf_to_ltf(prices, df_1h, volume_1h)
        
        # Volume spike: current 1h volume > 2.0x 20-period average
        vol_spike = volume_1h_aligned[i] > vol_ma_20_aligned[i] * 2.0
        
        # Regime filters
        is_trending = chop_1h_aligned[i] < 38.2   # Trending market
        is_ranging = chop_1h_aligned[i] > 61.8    # Ranging market
        
        # Entry logic: only enter when flat
        if position == 0:
            # In trending markets: trade breakouts with volume confirmation
            if is_trending:
                breakout_up = close[i] > high_20[i-1]  # Break above previous period's high
                breakout_down = close[i] < low_20[i-1]  # Break below previous period's low
                
                if breakout_up and vol_spike:
                    signals[i] = 0.25
                    position = 1
                    continue
                elif breakout_down and vol_spike:
                    signals[i] = -0.25
                    position = -1
                    continue
            
            # In ranging markets: trade mean reversion at Bollinger Bands
            elif is_ranging:
                # Get current 1h close for mean reversion signals
                close_1h_aligned = align_htf_to_ltf(prices, df_1h, close_1h)
                
                # Mean reversion: price touches Bollinger Band with volume spike
                if close_1h_aligned[i] <= bb_lower_aligned[i] and vol_spike:
                    signals[i] = 0.25  # Buy at lower BB
                    position = 1
                    continue
                elif close_1h_aligned[i] >= bb_upper_aligned[i] and vol_spike:
                    signals[i] = -0.25  # Sell at upper BB
                    position = -1
                    continue
        
        # Exit logic
        elif position == 1:
            # Exit long based on regime
            if is_trending:
                # In trending market: exit when price returns to middle of Donchian channel
                mid_channel = (high_20[i] + low_20[i]) / 2
                if close[i] < mid_channel:
                    signals[i] = 0.0
                    position = 0
                    continue
                else:
                    signals[i] = 0.25
            else:  # ranging market
                # In ranging market: exit when price returns to Bollinger Band middle
                close_1h_aligned = align_htf_to_ltf(prices, df_1h, close_1h)
                sma_20_aligned = align_htf_to_ltf(prices, df_1h, sma_20)
                if close_1h_aligned[i] >= sma_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Exit short based on regime
            if is_trending:
                # In trending market: exit when price returns to middle of Donchian channel
                mid_channel = (high_20[i] + low_20[i]) / 2
                if close[i] > mid_channel:
                    signals[i] = 0.0
                    position = 0
                    continue
                else:
                    signals[i] = -0.25
            else:  # ranging market
                # In ranging market: exit when price returns to Bollinger Band middle
                close_1h_aligned = align_htf_to_ltf(prices, df_1h, close_1h)
                sma_20_aligned = align_htf_to_ltf(prices, df_1h, sma_20)
                if close_1h_aligned[i] <= sma_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_ChopRegime_VolSpike_BreakoutMeanRev"
timeframe = "4h"
leverage = 1.0