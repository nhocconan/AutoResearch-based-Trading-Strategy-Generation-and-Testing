#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA50 trend filter and volume confirmation (>1.8x 30-period average)
    # Uses 1d EMA50 for primary trend direction to avoid counter-trend whipsaws in bear markets
    # Volume spike confirms institutional participation on breakouts
    # Exits on return to Camarilla pivot (H3/L3 midpoint) for mean reversion
    # Tight entry conditions target 25-40 trades/year (100-160 total over 4 years) to minimize fee drag
    # Works in bull markets via trend-following breakouts, in bear via faded counter-trend spikes
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate previous 4h bar's Camarilla levels (H3, L3)
    # H3 = close_prev + 1.1/4 * (high_prev - low_prev)
    # L3 = close_prev - 1.1/4 * (high_prev - low_prev)
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_h3 = prev_close + (1.1/4) * (prev_high - prev_low)
    camarilla_l3 = prev_close - (1.1/4) * (prev_high - prev_low)
    
    # Get 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 4h volume for confirmation (>1.8x 30-period average)
    vol_ma_4h = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma_4h[i] = np.mean(volume[i-30:i])
    volume_spike_4h = volume > (1.8 * vol_ma_4h)
    
    # Align all indicators to LTF (4h)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike_4h[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > camarilla_h3_aligned[i]
        short_breakout = close[i] < camarilla_l3_aligned[i]
        
        # 1d trend filter
        bullish_trend = close[i] > ema50_1d_aligned[i]
        bearish_trend = close[i] < ema50_1d_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike_4h[i]
        short_entry = short_breakout and bearish_trend and volume_spike_4h[i]
        
        # Exit logic: price returns to Camarilla pivot level (H3/L3 midpoint)
        camarilla_pivot = (camarilla_h3 + camarilla_l3) / 2
        camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot)
        
        # Exit when price returns to pivot level (within 0.3% tolerance)
        pivot_distance = abs(close[i] - camarilla_pivot_aligned[i]) / close[i]
        at_pivot = pivot_distance < 0.003
        
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

name = "4h_1d_camarilla_h3l3_ema50_volume_v1"
timeframe = "4h"
leverage = 1.0