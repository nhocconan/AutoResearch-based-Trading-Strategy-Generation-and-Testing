#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla H3/L3 breakout with 4h trend filter and volume confirmation
    # Trade breakouts aligned with 4h trend to avoid counter-trend whipsaws
    # Volume spike (>1.8x 20-period average) confirms institutional participation
    # Session filter (08-20 UTC) reduces noise trades
    # Designed for low frequency (target: 15-37/year) to minimize fee drag in 1h timeframe
    # Works in bull/bear markets by only trading with the dominant 4h trend
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculation (HTF for direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate previous 4h bar's Camarilla levels (H3, L3)
    # H3 = close_prev + 1.1 * (high_prev - low_prev) / 4
    # L3 = close_prev - 1.1 * (high_prev - low_prev) / 4
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    # Get 4h EMA20 for trend filter
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Get 4h volume for confirmation (>1.8x 20-period average)
    vol_ma_4h = np.full(len(df_4h), np.nan)
    for i in range(20, len(df_4h)):
        vol_ma_4h[i] = np.mean(volume_4h[i-20:i])
    volume_spike_4h = volume_4h > (1.8 * vol_ma_4h)
    
    # Align all indicators to LTF (1h)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    volume_spike_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h)
    
    # Session filter: 08-20 UTC (pre-compute hours array)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema20_4h_aligned[i]) or np.isnan(volume_spike_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > camarilla_h3_aligned[i]
        short_breakout = close[i] < camarilla_l3_aligned[i]
        
        # 4h trend filter
        bullish_trend = close[i] > ema20_4h_aligned[i]
        bearish_trend = close[i] < ema20_4h_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = False
        short_entry = False
        
        # Long: bullish breakout above H3 + bullish 4h trend + volume spike
        if long_breakout and bullish_trend:
            long_entry = volume_spike_aligned[i]
        # Short: bearish breakout below L3 + bearish 4h trend + volume spike
        elif short_breakout and bearish_trend:
            short_entry = volume_spike_aligned[i]
        
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

name = "1h_4h_camarilla_h3l3_trend_volume_session_v1"
timeframe = "1h"
leverage = 1.0