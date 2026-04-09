#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation
# - Primary timeframe: 1d (daily candles)
# - HTF: 1w (weekly) for trend direction using HMA(21)
# - Entry: Long when price breaks above Donchian(20) upper band AND weekly HMA trending up
#          Short when price breaks below Donchian(20) lower band AND weekly HMA trending down
# - Volume confirmation: Current volume > 1.5x 20-day average volume
# - ATR-based stoploss: Exit when price moves 2.5x ATR against position
# - Fixed position size: 0.25 to control drawdown
# - Works in both bull and bear markets by requiring weekly trend alignment
# - Target: 15-25 trades/year on 1d timeframe (60-100 total over 4 years)

name = "1d_1w_donchian_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w HMA(21) for trend direction
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    # Pad arrays for convolution
    close_padded = np.concatenate([np.full(half_len, close_1w[0]), close_1w])
    wma_half = wma(close_padded, half_len)
    
    close_padded_full = np.concatenate([np.full(21, close_1w[0]), close_1w])
    wma_full = wma(close_padded_full, 21)
    
    # Align lengths
    min_len = min(len(wma_half), len(wma_full))
    wma_half = wma_half[-min_len:]
    wma_full = wma_full[-min_len:]
    
    hma_raw = 2 * wma_half - wma_full
    hma_1w = wma(hma_raw, sqrt_len)
    
    # Handle padding
    hma_1w = np.concatenate([np.full(len(close_1w) - len(hma_1w), np.nan), hma_1w])
    
    # Determine trend: slope of HMA over 3 periods
    hma_slope = np.diff(hma_1w, prepend=hma_1w[0])
    hma_trending_up = hma_slope > 0
    hma_trending_down = hma_slope < 0
    
    # Align HTF indicators to 1d timeframe
    hma_trending_up_aligned = align_htf_to_ltf(prices, df_1w, hma_trending_up)
    hma_trending_down_aligned = align_htf_to_ltf(prices, df_1w, hma_trending_down)
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    
    # Calculate ATR (20-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(atr[i]) or atr[i] <= 0 or
            np.isnan(hma_trending_up_aligned[i]) or np.isnan(hma_trending_down_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # ATR-based trailing stop: exit if price drops 2.5x ATR from highest high
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # ATR-based trailing stop: exit if price rises 2.5x ATR from lowest low
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check volume confirmation first
            if volume_confirm[i]:
                # Long entry: price breaks above Donchian upper AND weekly HMA trending up
                if close[i] > donchian_upper[i] and hma_trending_up_aligned[i]:
                    position = 1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian lower AND weekly HMA trending down
                elif close[i] < donchian_lower[i] and hma_trending_down_aligned[i]:
                    position = -1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = -0.25
    
    return signals