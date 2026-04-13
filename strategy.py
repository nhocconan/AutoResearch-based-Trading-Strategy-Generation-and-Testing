#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla H3/L3 mean reversion with volume confirmation and 1d trend filter
    # Designed to capture reversions to the mean at key daily levels with volume confirmation
    # Target: 100-150 trades over 4 years (25-38/year) for low fee drag and good generalization
    # Works in both bull and bear markets by trading mean reversion at Camarilla levels
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values if 'volume' in df_4h.columns else np.ones(len(df_4h))
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # Camarilla levels (H3/L3 are the key mean reversion levels)
    camarilla_h3 = prev_close_1d + 1.125 * (prev_high_1d - prev_low_1d)
    camarilla_l3 = prev_close_1d - 1.125 * (prev_high_1d - prev_low_1d)
    
    # Calculate 1d EMA50 for trend filter (more responsive than EMA200)
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 4h primary timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume_4h[i] > 1.5 * vol_avg_20_aligned[i]
        
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

name = "4h_1d_camarilla_meanreversion_volume_trend_v1"
timeframe = "4h"
leverage = 1.0