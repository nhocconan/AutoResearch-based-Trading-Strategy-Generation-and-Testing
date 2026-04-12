#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla H3/L3 breakout with 4h trend filter (EMA34) and volume confirmation
    # Uses 4h for signal direction and 1d Camarilla for structure to reduce noise
    # Session filter (08-20 UTC) and discrete position sizing (0.20) to control trade frequency
    # Target: 15-25 trades/year (60-100 total) to minimize fee drag
    # Works in bull/bear by only trading with dominant 4h trend
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate previous 4h bar's Camarilla H3/L3 for breakout levels
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    prev_close_4h[0] = np.nan
    
    camarilla_h3_4h = prev_close_4h + 1.1 * (prev_high_4h - prev_low_4h) / 4
    camarilla_l3_4h = prev_close_4h - 1.1 * (prev_high_4h - prev_low_4h) / 4
    
    # Get 4h EMA34 for trend filter
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 4h volume confirmation (>1.3x 20-period average)
    vol_ma_4h = np.full(len(df_4h), np.nan)
    for i in range(20, len(df_4h)):
        vol_ma_4h[i] = np.mean(volume_4h[i-20:i])
    volume_spike_4h = volume_4h > (1.3 * vol_ma_4h)
    
    # Get 1d data for Camarilla pivot (mean reversion exit)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot for exit
    camarilla_pivot_1d = (high_1d + low_1d + close_1d) / 3
    
    # Align all indicators to LTF (1h)
    camarilla_h3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3_4h)
    camarilla_l3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3_4h)
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    volume_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h)
    camarilla_pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_h3_4h_aligned[i]) or np.isnan(camarilla_l3_4h_aligned[i]) or 
            np.isnan(ema34_4h_aligned[i]) or np.isnan(volume_spike_4h_aligned[i]) or
            np.isnan(camarilla_pivot_1d_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > camarilla_h3_4h_aligned[i]
        short_breakout = close[i] < camarilla_l3_4h_aligned[i]
        
        # 4h trend filter
        bullish_trend = close[i] > ema34_4h_aligned[i]
        bearish_trend = close[i] < ema34_4h_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike_4h_aligned[i]
        short_entry = short_breakout and bearish_trend and volume_spike_4h_aligned[i]
        
        # Exit logic: price returns to 1d Camarilla pivot (mean reversion)
        pivot_distance = abs(close[i] - camarilla_pivot_1d_aligned[i]) / close[i]
        at_pivot = pivot_distance < 0.0015  # 0.15% tolerance
        
        long_exit = at_pivot or not bullish_trend
        short_exit = at_pivot or not bearish_trend
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_camarilla_h3l3_ema34_volume_v1"
timeframe = "1h"
leverage = 1.0