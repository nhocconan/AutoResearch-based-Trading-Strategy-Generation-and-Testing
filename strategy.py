#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla breakout with 12h volume confirmation and 1d trend filter
    # Designed for low trade frequency (12-37/year) to minimize fee drag on 6h timeframe
    # Uses 12h/1d for signal direction and confirmation, 6h only for execution
    # Works in both bull and bear: breakout continuation in trend, mean reversion in range
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 6h data for primary calculations
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values if 'volume' in df_6h.columns else np.ones(len(df_6h))
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values if 'volume' in df_12h.columns else np.ones(len(df_12h))
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 6h Camarilla pivot levels (based on previous 6h bar)
    prev_high_6h = np.roll(high_6h, 1)
    prev_low_6h = np.roll(low_6h, 1)
    prev_close_6h = np.roll(close_6h, 1)
    
    # Camarilla levels
    camarilla_h3 = prev_close_6h + 1.125 * (prev_high_6h - prev_low_6h)  # H3: strong resistance
    camarilla_l3 = prev_close_6h - 1.125 * (prev_high_6h - prev_low_6h)  # L3: strong support
    camarilla_h4 = prev_close_6h + 1.5 * (prev_high_6h - prev_low_6h)    # H4: breakout level
    camarilla_l4 = prev_close_6h - 1.5 * (prev_high_6h - prev_low_6h)    # L4: breakout level
    
    # Calculate 12h volume average (20-period)
    vol_avg_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d ATR for volatility filter
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align all HTF indicators to 6h primary timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_6h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_6h, camarilla_l4)
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(vol_avg_20_12h_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.8x 20-period average
        # Get the 12h bar index for current 6h bar (each 12h bar = 2 6h bars)
        idx_12h = i // 2
        if idx_12h >= len(volume_12h):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_12h[idx_12h] > 1.8 * vol_avg_20_12h_aligned[i]
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr_1d_aligned[i] > 0.01 * close[i]  # ATR > 1% of price
        
        # Breakout conditions at Camarilla H4/L4 levels
        breakout_long = close[i] > camarilla_h4_aligned[i] and close[i-1] <= camarilla_h4_aligned[i-1]
        breakout_short = close[i] < camarilla_l4_aligned[i] and close[i-1] >= camarilla_l4_aligned[i-1]
        
        # Mean reversion conditions at Camarilla H3/L3 levels (fade extreme moves)
        reversion_long = close[i] < camarilla_l3_aligned[i] and close[i-1] >= camarilla_l3_aligned[i-1]
        reversion_short = close[i] > camarilla_h3_aligned[i] and close[i-1] <= camarilla_h3_aligned[i-1]
        
        # Trend filter: only trade breakouts in direction of 1d EMA50
        # For long breakouts: price above EMA50; for short breakouts: price below EMA50
        trend_filter_long = close[i] > ema50_1d_aligned[i]
        trend_filter_short = close[i] < ema50_1d_aligned[i]
        
        # Entry conditions
        enter_long = (breakout_long and volume_confirmed and vol_filter and trend_filter_long) or \
                     (reversion_long and volume_confirmed and vol_filter and not trend_filter_long)
        enter_short = (breakout_short and volume_confirmed and vol_filter and trend_filter_short) or \
                      (reversion_short and volume_confirmed and vol_filter and not trend_filter_short)
        
        # Exit conditions: opposite Camarilla level or volatility expansion
        exit_long = position == 1 and (close[i] < camarilla_l3_aligned[i] or atr_1d_aligned[i] > 2.5 * atr_1d_aligned[max(0, i-1)])
        exit_short = position == -1 and (close[i] > camarilla_h3_aligned[i] or atr_1d_aligned[i] > 2.5 * atr_1d_aligned[max(0, i-1)])
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0