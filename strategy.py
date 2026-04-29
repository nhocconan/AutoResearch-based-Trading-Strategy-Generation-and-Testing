#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Long when price breaks above 6h Donchian upper AND weekly pivot > previous week pivot (bullish bias) AND volume > 2x average
# Short when price breaks below 6h Donchian lower AND weekly pivot < previous week pivot (bearish bias) AND volume > 2x average
# Exit on opposite Donchian break or weekly pivot flip
# Uses proven Donchian structure with weekly pivot bias for BTC/ETH in both bull/bear markets
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag

name = "6h_Donchian20_1wPivot_Direction_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for weekly calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot point (typical price) for trend bias
    # Pivot = (high + low + close) / 3
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    weekly_pivot = typical_price.values
    
    # Align weekly pivot to 6h timeframe (completed weekly bar only)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate 6h Donchian channels (20-period)
    # Upper = max(high, 20), Lower = min(low, 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # warmup for Donchian and weekly pivot
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(weekly_pivot_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        curr_pivot = weekly_pivot_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Determine weekly pivot trend: compare with previous week's pivot
        # Need to get previous aligned pivot value (requires looking back ~28 bars for 6h->1w)
        if i >= 28:  # ~4 weeks * 28 bars/week (6h bars per week)
            prev_pivot_idx = i - 28
            if prev_pivot_idx >= 0 and not np.isnan(weekly_pivot_aligned[prev_pivot_idx]):
                prev_pivot = weekly_pivot_aligned[prev_pivot_idx]
                is_bullish_bias = curr_pivot > prev_pivot
                is_bearish_bias = curr_pivot < prev_pivot
            else:
                # Not enough data for pivot comparison, stay flat
                signals[i] = 0.0
                continue
        else:
            # Not enough data for pivot trend, stay flat until warmup complete
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: price breaks above Donchian upper AND bullish weekly pivot bias
                if curr_high > curr_upper and is_bullish_bias:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below Donchian lower AND bearish weekly pivot bias
                elif curr_low < curr_lower and is_bearish_bias:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price breaks below Donchian lower OR weekly pivot turns bearish
            if curr_low < curr_lower or not is_bullish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price breaks above Donchian upper OR weekly pivot turns bullish
            if curr_high > curr_upper or not is_bearish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals