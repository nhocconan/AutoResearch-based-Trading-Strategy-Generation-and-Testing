#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter (HMA21) and volume confirmation
# Uses Camarilla pivot levels from 4h data: breakout above H3 = long, below L3 = short
# 4h HMA21 filter ensures trades align with higher timeframe trend
# Volume confirmation reduces false breakouts
# Designed for 1h timeframe to target 60-150 total trades over 4 years (15-37/year)
# Session filter (08-20 UTC) reduces noise trades
# Works in bull/bear: HMA21 adapts to trend, Camarilla provides robust support/resistance

name = "1h_4h_camarilla_hma_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Camarilla pivots and HMA21
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot levels (based on previous day's OHLC)
    # For intraday, we use the previous 4h bar's OHLC as proxy for "yesterday"
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels: based on previous period's range
    # H3 = close + 1.1*(high-low)/2
    # L3 = close - 1.1*(high-low)/2
    # H4 = close + 1.1*(high-low)
    # L4 = close - 1.1*(high-low)
    # We'll use H3/L3 for entries, H4/L4 for stops (but we'll use trend for exits)
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    range_4h = prev_high - prev_low
    h3 = prev_close + 1.1 * range_4h / 2
    l3 = prev_close - 1.1 * range_4h / 2
    h4 = prev_close + 1.1 * range_4h
    l4 = prev_close - 1.1 * range_4h
    
    # Align 4h Camarilla levels to 1h timeframe
    h3_1h = align_htf_to_ltf(prices, df_4h, h3)
    l3_1h = align_htf_to_ltf(prices, df_4h, l3)
    h4_1h = align_htf_to_ltf(prices, df_4h, h4)
    l4_1h = align_htf_to_ltf(prices, df_4h, l4)
    
    # Calculate 4h HMA21 trend filter
    half_n = int(21/2 + 0.5)
    wma_half = pd.Series(close_4h).rolling(window=half_n, min_periods=half_n).mean()
    wma_full = pd.Series(close_4h).rolling(window=21, min_periods=21).mean()
    hma_21_4h = (2 * wma_half - wma_full).values
    
    # Align 4h HMA21 to 1h timeframe
    hma_21_1h = align_htf_to_ltf(prices, df_4h, hma_21_4h)
    
    # Calculate 20-period average volume for volume confirmation (1h volume)
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid or outside session
        if (np.isnan(h3_1h[i]) or np.isnan(l3_1h[i]) or
            np.isnan(hma_21_1h[i]) or np.isnan(avg_volume[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Camarilla L3 OR trend turns bearish
            if close[i] < l3_1h[i] or close[i] < hma_21_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla H3 OR trend turns bullish
            if close[i] > h3_1h[i] or close[i] > hma_21_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Entry logic with volume confirmation and session filter
            if volume_confirm:
                # Long breakout: price closes above Camarilla H3 AND price > 4h HMA21 (bullish trend)
                if close[i] > h3_1h[i] and close[i] > hma_21_1h[i]:
                    position = 1
                    signals[i] = 0.20
                # Short breakout: price closes below Camarilla L3 AND price < 4h HMA21 (bearish trend)
                elif close[i] < l3_1h[i] and close[i] < hma_21_1h[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals