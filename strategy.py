#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla H4/L4 breakout with 12h trend filter and volume confirmation
    # Trade breakouts aligned with 12h trend to avoid counter-trend whipsaws
    # Volume spike (>2.0x 20-period average) confirms institutional participation
    # Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
    # Works in bull/bear markets by only trading with the dominant 12h trend
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla calculation and trend filter (HTF for direction)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate previous 12h bar's Camarilla levels (H4, L4)
    # H4 = close_prev + 1.1 * (high_prev - low_prev) / 2
    # L4 = close_prev - 1.1 * (high_prev - low_prev) / 2
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Get 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 12h volume for confirmation (>2.0x 20-period average)
    vol_ma_12h = np.full(len(df_12h), np.nan)
    for i in range(20, len(df_12h)):
        vol_ma_12h[i] = np.mean(volume_12h[i-20:i])
    volume_spike_12h = volume_12h > (2.0 * vol_ma_12h)
    
    # Align all indicators to LTF (6h)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > camarilla_h4_aligned[i]
        short_breakout = close[i] < camarilla_l4_aligned[i]
        
        # 12h trend filter
        bullish_trend = close[i] > ema50_12h_aligned[i]
        bearish_trend = close[i] < ema50_12h_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = False
        short_entry = False
        
        # Long: bullish breakout above H4 + bullish 12h trend + volume spike
        if long_breakout and bullish_trend:
            long_entry = volume_spike_aligned[i]
        # Short: bearish breakout below L4 + bearish 12h trend + volume spike
        elif short_breakout and bearish_trend:
            short_entry = volume_spike_aligned[i]
        
        # Exit logic: price returns to Camarilla pivot level (mean reversion)
        # Camarilla pivot = (high_prev + low_prev + close_prev) / 3
        camarilla_pivot = (prev_high + prev_low + prev_close) / 3
        camarilla_pivot_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pivot)
        
        # Exit when price returns to pivot level (within 0.1% tolerance)
        pivot_distance = abs(close[i] - camarilla_pivot_aligned[i]) / close[i]
        at_pivot = pivot_distance < 0.001
        
        long_exit = at_pivot or not bullish_trend
        short_exit = at_pivot or not bearish_trend
        
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

name = "6h_12h_camarilla_h4l4_trend_volume_v1"
timeframe = "6h"
leverage = 1.0