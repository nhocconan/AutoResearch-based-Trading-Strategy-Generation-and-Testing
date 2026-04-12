#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with 1d trend filter and volume confirmation
    # Camarilla levels (H3/L3 for mean reversion, H4/L4 for breakout) from 1d pivot
    # Only trade breakouts (H4/L4) in direction of 1d EMA200 trend
    # Volume confirmation (>2.0x 20-period average) reduces false breakouts
    # Designed for low trade frequency (target: 12-37/year) to minimize fee drag
    # Works in bull/bear markets by only trading strong aligned breakouts
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla pivot levels (using previous day's OHLC)
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, L4 = close - 1.1*(high-low)*1.1/2
    # Simplified: H4 = close + 1.1*(high-low), L4 = close - 1.1*(high-low)
    # Actually: H3 = close + 1.1*(high-low)/2, H4 = close + 1.1*(high-low)
    #          L3 = close - 1.1*(high-low)/2, L4 = close - 1.1*(high-low)
    hl_range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + 1.1 * hl_range_1d
    camarilla_l4 = close_1d - 1.1 * hl_range_1d
    camarilla_h3 = close_1d + 1.1 * hl_range_1d / 2
    camarilla_l3 = close_1d - 1.1 * hl_range_1d / 2
    
    # 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # 1d volume confirmation: volume > 2.0 * 20-period average
    vol_ma_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    
    # Align all indicators to LTF (6h)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions: price breaks H4 or L4 with volume confirmation
        bullish_breakout = (close[i] > camarilla_h4_aligned[i]) and volume_spike_aligned[i]
        bearish_breakout = (close[i] < camarilla_l4_aligned[i]) and volume_spike_aligned[i]
        
        # 1d trend filter: only trade breakouts in direction of higher timeframe trend
        bullish_trend = close[i] > ema200_1d_aligned[i]
        bearish_trend = close[i] < ema200_1d_aligned[i]
        
        # Mean reversion fade at H3/L3 (optional counter-trend entries)
        # Fade at H3: short when price reaches H3 in uptrend, expecting pullback
        # Fade at L3: long when price reaches L3 in downtrend, expecting bounce
        fade_short = (close[i] >= camarilla_h3_aligned[i]) and bullish_trend and volume_spike_aligned[i]
        fade_long = (close[i] <= camarilla_l3_aligned[i]) and bearish_trend and volume_spike_aligned[i]
        
        # Entry logic
        long_entry = False
        short_entry = False
        
        # Long: bullish breakout OR fade at L3 in downtrend
        if bullish_breakout and bullish_trend:
            long_entry = True
        elif fade_long:
            long_entry = True
            
        # Short: bearish breakout OR fade at H3 in uptrend
        if bearish_breakout and bearish_trend:
            short_entry = True
        elif fade_short:
            short_entry = True
        
        # Exit logic: reverse signal or price returns to pivot levels
        long_exit = bearish_breakout or (close[i] < camarilla_h3_aligned[i]) or (close[i] > camarilla_l4_aligned[i] and position == 1)
        short_exit = bullish_breakout or (close[i] > camarilla_l3_aligned[i]) or (close[i] < camarilla_h4_aligned[i] and position == -1)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_camarilla_breakout_fade_v1"
timeframe = "6h"
leverage = 1.0