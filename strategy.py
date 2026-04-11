#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d + volume spike + chop regime filter
# - Long when price touches or crosses above Camarilla H4 level (strong resistance turned support)
# - Short when price touches or crosses below Camarilla L4 level (strong support turned resistance)
# - Volume confirmation: current volume > 1.8x 20-period average (using 1d aligned volume)
# - Chop regime filter: only trade when Choppiness Index (14) > 61.8 (range-bound market) for mean reversion
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 12h
# - Works in both bull (mean reversion in range) and bear (mean reversion in range) markets
# - 1d HTF provides reliable Camarilla levels, reducing false signals from lower timeframe noise

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
    
    # Load 1d data ONCE before loop for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # H4 = Close + 1.5*(High - Low)
    # L4 = Close - 1.5*(High - Low)
    camarilla_h4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_l4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Pre-compute 1d volume SMA (20-period)
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    volume_sma_20_1d = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute Choppiness Index (14) on 1d
    # TR = max(H-L, abs(H-PC), abs(L-PC))
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to 0 (no previous close)
    tr_1d[0] = 0.0
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    # True range of high-low over 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # Choppiness Index = 100 * log10(sum_TR_14 / (max_high_14 - min_low_14)) / log10(14)
    # Avoid division by zero
    denominator = max_high_14 - min_low_14
    chop_14_1d = np.where(
        denominator > 0,
        100 * np.log10(tr_sum_14 / denominator) / np.log10(14),
        50.0  # neutral value when no range
    )
    
    # Align 1d indicators to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    chop_14_aligned = align_htf_to_ltf(prices, df_1d, chop_14_1d)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(chop_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Camarilla touch conditions (with small buffer to avoid whipsaw)
        # Using 0.1% buffer around levels
        buffer = 0.001
        touch_long = price_low <= camarilla_h4_aligned[i] * (1 + buffer) and price_high >= camarilla_h4_aligned[i] * (1 - buffer)
        touch_short = price_high >= camarilla_l4_aligned[i] * (1 - buffer) and price_low <= camarilla_l4_aligned[i] * (1 + buffer)
        
        # Volume confirmation: current volume > 1.8x 20-period average (using 1d aligned volume)
        vol_confirm = volume_current > 1.8 * volume_sma_20_aligned[i]
        
        # Chop regime filter: only trade when Chop > 61.8 (range-bound market)
        chop_filter = chop_14_aligned[i] > 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Touch Camarilla H4 (resistance turned support) + volume + chop
        if touch_long and vol_confirm and chop_filter:
            enter_long = True
        
        # Short: Touch Camarilla L4 (support turned resistance) + volume + chop
        if touch_short and vol_confirm and chop_filter:
            enter_short = True
        
        # Exit conditions: opposite Camarilla touch or chop regime change
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price touches L4 OR chop drops below 61.8 (trending)
            exit_long = touch_short or (not chop_filter)
        elif position == -1:
            # Exit short if price touches H4 OR chop drops below 61.8 (trending)
            exit_exit_short = touch_long or (not chop_filter)
            exit_short = exit_exit_short
        
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