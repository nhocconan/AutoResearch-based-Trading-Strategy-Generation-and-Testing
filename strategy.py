#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h mean reversion at daily Camarilla H3/L3 with daily volume spike and 1d trend filter
    # Designed for low trade frequency (12-37/year) to minimize fee drag on 12h timeframe
    # Uses 1d for signal direction, 12h only for entry timing precision
    # Works in both bull and bear: mean reversion in range, trend filter avoids counter-trend trades
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 1d Camarilla pivot levels (based on previous 1d bar)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # Camarilla levels (H3/L3 are the key mean reversion levels)
    camarilla_h3 = prev_close_1d + 1.125 * (prev_high_1d - prev_low_1d)
    camarilla_l3 = prev_close_1d - 1.125 * (prev_high_1d - prev_low_1d)
    
    # Calculate 1d volume average (20-period)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 4h data for trend filter (faster than 1d for trend confirmation)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA20 for trend filter
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all HTF indicators to 12h primary timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(ema20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        # Get the 1d bar index for current 12h bar (each 1d bar = 2 12h bars)
        idx_1d = i // 2
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_1d[idx_1d] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Mean reversion conditions at Camarilla H3/L3 levels
        reversion_long = close[i] < camarilla_l3_aligned[i]  # Price below L3 -> long (expect bounce up)
        reversion_short = close[i] > camarilla_h3_aligned[i]  # Price above H3 -> short (expect bounce down)
        
        # Trend filter: only trade in direction of 4h EMA20
        # For long: price above EMA20; for short: price below EMA20
        trend_filter_long = close[i] > ema20_4h_aligned[i]
        trend_filter_short = close[i] < ema20_4h_aligned[i]
        
        # Entry conditions
        enter_long = reversion_long and volume_confirmed and trend_filter_long
        enter_short = reversion_short and volume_confirmed and trend_filter_short
        
        # Exit conditions: price returns to Camarilla H4/L4 levels (stronger levels)
        camarilla_h4 = prev_close_1d + 1.5 * (prev_high_1d - prev_low_1d)
        camarilla_l4 = prev_close_1d - 1.5 * (prev_high_1d - prev_low_1d)
        camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
        camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
        
        exit_long = position == 1 and close[i] >= camarilla_h4_aligned[i]
        exit_short = position == -1 and close[i] <= camarilla_l4_aligned[i]
        
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

name = "12h_1d_camarilla_meanreversion_volume_trend_v1"
timeframe = "12h"
leverage = 1.0