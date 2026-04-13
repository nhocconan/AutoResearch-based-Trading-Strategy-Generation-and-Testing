#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d mean reversion at weekly Camarilla H3/L3 with weekly volume spike and 1w trend filter
    # Designed for very low trade frequency (7-25/year) to minimize fee drag on 1d timeframe
    # Uses 1w for signal direction, 1d only for entry timing precision
    # Works in both bull and bear: mean reversion in range, trend filter avoids counter-trend trades
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1w data for HTF Camarilla levels and volume
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values if 'volume' in df_1w.columns else np.ones(len(df_1w))
    
    # Calculate 1w Camarilla pivot levels (based on previous 1w bar)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    
    # Camarilla levels (H3/L3 are the key mean reversion levels)
    camarilla_h3 = prev_close_1w + 1.125 * (prev_high_1w - prev_low_1w)
    camarilla_l3 = prev_close_1w - 1.125 * (prev_high_1w - prev_low_1w)
    
    # Calculate 1w volume average (20-period)
    vol_avg_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all HTF indicators to 1d primary timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    vol_avg_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20_1w)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_avg_20_1w_aligned[i]) or
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1w volume > 1.5x 20-period average
        # Get the 1w bar index for current 1d bar (each 1w bar = 7 1d bars)
        idx_1w = i // 7
        if idx_1w >= len(volume_1w):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_1w[idx_1w] > 1.5 * vol_avg_20_1w_aligned[i]
        
        # Mean reversion conditions at Camarilla H3/L3 levels
        reversion_long = close[i] < camarilla_l3_aligned[i]  # Price below L3 -> long (expect bounce up)
        reversion_short = close[i] > camarilla_h3_aligned[i]  # Price above H3 -> short (expect bounce down)
        
        # Trend filter: only trade in direction of 1d EMA50
        # For long: price above EMA50; for short: price below EMA50
        trend_filter_long = close[i] > ema50_1d_aligned[i]
        trend_filter_short = close[i] < ema50_1d_aligned[i]
        
        # Entry conditions
        enter_long = reversion_long and volume_confirmed and trend_filter_long
        enter_short = reversion_short and volume_confirmed and trend_filter_short
        
        # Exit conditions: price returns to Camarilla H4/L4 levels (stronger levels)
        camarilla_h4 = prev_close_1w + 1.5 * (prev_high_1w - prev_low_1w)
        camarilla_l4 = prev_close_1w - 1.5 * (prev_high_1w - prev_low_1w)
        camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
        camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
        
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

name = "1d_1w_camarilla_meanreversion_volume_trend_v1"
timeframe = "1d"
leverage = 1.0