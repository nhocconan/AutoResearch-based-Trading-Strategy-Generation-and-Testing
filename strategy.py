#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return signals
    
    # Calculate 1d OHLC for Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Calculate Camarilla levels (based on previous day's range)
    # R4 = Close + ((High - Low) * 1.1/2)
    # R3 = Close + ((High - Low) * 1.1/4)
    # R2 = Close + ((High - Low) * 1.1/6)
    # R1 = Close + ((High - Low) * 1.1/12)
    # S1 = Close - ((High - Low) * 1.1/12)
    # S2 = Close - ((High - Low) * 1.1/6)
    # S3 = Close - ((High - Low) * 1.1/4)
    # S4 = Close - ((High - Low) * 1.1/2)
    daily_range = high_1d - low_1d
    
    camarilla_r4 = close_1d + (daily_range * 1.1 / 2)
    camarilla_r3 = close_1d + (daily_range * 1.1 / 4)
    camarilla_r2 = close_1d + (daily_range * 1.1 / 6)
    camarilla_r1 = close_1d + (daily_range * 1.1 / 12)
    camarilla_s1 = close_1d - (daily_range * 1.1 / 12)
    camarilla_s2 = close_1d - (daily_range * 1.1 / 6)
    camarilla_s3 = close_1d - (daily_range * 1.1 / 4)
    camarilla_s4 = close_1d - (daily_range * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: 20-period average on 4h
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Trend filter: 50-period EMA on 4h
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(ema_50[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Trend filter: price above EMA50 for long, below EMA50 for short
        trend_up = price_close > ema_50[i]
        trend_down = price_close < ema_50[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price breaks above Camarilla R4 level + volume confirmation + uptrend
        if price_close > camarilla_r4_aligned[i] and vol_confirm and trend_up:
            enter_long = True
        
        # Short: Price breaks below Camarilla S4 level + volume confirmation + downtrend
        if price_close < camarilla_s4_aligned[i] and vol_confirm and trend_down:
            enter_short = True
        
        # Exit conditions: price returns to the Camarilla midpoint (close of previous day)
        camarilla_midpoint = (camarilla_r4_aligned[i] + camarilla_s4_aligned[i]) / 2
        exit_long = price_close < camarilla_midpoint
        exit_short = price_close > camarilla_midpoint
        
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

# Hypothesis: Camarilla breakout on daily timeframe with volume confirmation and trend filter.
# Uses 1d Camarilla pivot levels (R4/S4) for breakout entries and midpoint for exits.
# Volume confirmation (>1.5x 20-period average) ensures institutional participation.
# Trend filter (50-period EMA) ensures we trade in the direction of the 4h trend.
# Works in both bull and breakout scenarios by capturing institutional breakouts.
# Reduced position size to 0.25 to manage risk. Target: 20-40 trades/year to minimize fee drag.