#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d + volume confirmation + chop regime filter
# - Uses 1d Camarilla pivot levels (H3/L3 for entries, H4/L4 for stops)
# - Entry when price touches H3/L3 with volume > 1.5x 20-period average
# - Only trade in ranging markets (choppiness index > 61.8 on 12h)
# - Exit when price reaches opposite H3/L3 level or volume dries up
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Camarilla pivots work well in ranging markets which are common in bear/consolidation periods
# - Volume confirmation ensures institutional participation
# - Chop filter avoids false signals during strong trends

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Camarilla pivots and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and ranges
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    H3 = pivot + (range_1d * 1.1 / 4)
    L3 = pivot - (range_1d * 1.1 / 4)
    H4 = pivot + (range_1d * 1.1 / 2)
    L4 = pivot - (range_1d * 1.1 / 2)
    
    # Align 1d levels to 12h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Pre-compute 12h Choppiness Index (chop > 61.8 = ranging market)
    # Chop = 100 * log10(sum(ATR) / log10(highest_high - lowest_low)) / log10(n)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # True Range
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Sum of ATR over 14 periods
    sum_atr = atr.rolling(window=14, min_periods=14).sum()
    
    # Highest high and lowest low over 14 periods
    highest_high = high_series.rolling(window=14, min_periods=14).max()
    lowest_low = low_series.rolling(window=14, min_periods=14).min()
    range_14 = highest_high - lowest_low
    
    # Choppiness Index
    chop = 100 * np.log10(sum_atr / range_14) / np.log10(14)
    chop_values = chop.fillna(50).values  # Fill NaN with neutral value
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Chop regime filter: only trade in ranging markets (chop > 61.8)
        chop_filter = chop_aligned[i] > 61.8
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Entry conditions: price touches H3/L3 with volume and chop confirmation
        # Use high/low of bar to capture touches that might not close at the level
        touch_H3 = price_high >= H3_aligned[i] and price_low <= H3_aligned[i]
        touch_L3 = price_high >= L3_aligned[i] and price_low <= L3_aligned[i]
        
        enter_long = False
        enter_short = False
        
        # Long: price touches L3 (support) with volume confirmation in ranging market
        if touch_L3 and vol_confirm and chop_filter:
            enter_long = True
        
        # Short: price touches H3 (resistance) with volume confirmation in ranging market
        if touch_H3 and vol_confirm and chop_filter:
            enter_short = True
        
        # Exit conditions: price reaches opposite H3/L3 level or volume/chop deteriorates
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price reaches H3 (resistance) OR chop breaks down OR volume dries up
            exit_long = (price_high >= H3_aligned[i]) or (chop_aligned[i] <= 50) or (volume_current < volume_sma_20[i])
        elif position == -1:
            # Exit short if price reaches L3 (support) OR chop breaks down OR volume dries up
            exit_short = (price_low <= L3_aligned[i]) or (chop_aligned[i] <= 50) or (volume_current < volume_sma_20[i])
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals