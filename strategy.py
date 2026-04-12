#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume confirmation
    # Uses 4h for signal direction (reduces noise), 1h for precise entry timing
    # Volume spike (>2.0x 20-period average) confirms institutional participation
    # Session filter (08-20 UTC) reduces low-liquidity noise trades
    # Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag
    # Works in bull/bear markets by only trading with the dominant 4h trend
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculation and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate previous 4h bar's Camarilla levels (H3, L3)
    # H3 = close_prev + 1.1 * (high_prev - low_prev) / 2
    # L3 = close_prev - 1.1 * (high_prev - low_prev) / 2
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Get 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1h volume for confirmation (>2.0x 20-period average)
    vol_ma_1h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_1h[i] = np.mean(volume[i-20:i])
    volume_spike_1h = volume > (2.0 * vol_ma_1h)
    
    # Align all indicators to LTF (1h)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_spike_1h[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > camarilla_h3_aligned[i]
        short_breakout = close[i] < camarilla_l3_aligned[i]
        
        # 4h trend filter
        bullish_trend = close[i] > ema50_4h_aligned[i]
        bearish_trend = close[i] < ema50_4h_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike_1h[i]
        short_entry = short_breakout and bearish_trend and volume_spike_1h[i]
        
        # Exit logic: price returns to Camarilla pivot level (mean reversion)
        # Camarilla pivot = (high_prev + low_prev + close_prev) / 3
        camarilla_pivot = (prev_high + prev_low + prev_close) / 3
        camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot)
        
        # Exit when price returns to pivot level (within 0.1% tolerance)
        pivot_distance = abs(close[i] - camarilla_pivot_aligned[i]) / close[i]
        at_pivot = pivot_distance < 0.001
        
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

name = "1h_4h_camarilla_h3l3_ema50_volume_v1"
timeframe = "1h"
leverage = 1.0