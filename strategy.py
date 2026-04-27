#!/usr/bin/env python3
"""
6h_OrderBlock_Breakout_12hTrend_VolumeSpike
Hypothesis: Institutional order blocks on 12h act as support/resistance. Breakouts confirmed by 12h trend and volume spikes capture institutional flow. Works in bull/bear as order blocks persist across regimes. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for order blocks and trend
    df_12h = get_htf_data(prices, '12h')
    
    # Identify bullish/bearish order blocks on 12h
    # Bullish OB: last down candle before up move (close < open, then next candle closes above its high)
    # Bearish OB: last up candle before down move (close > open, then next candle closes below its low)
    open_12h = df_12h['open'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    bullish_ob = np.zeros(len(close_12h), dtype=bool)
    bearish_ob = np.zeros(len(close_12h), dtype=bool)
    
    for i in range(1, len(close_12h)-1):
        # Bullish OB: candle i is bearish (close < open), candle i+1 is bullish and closes > high[i]
        if close_12h[i] < open_12h[i] and close_12h[i+1] > open_12h[i+1] and close_12h[i+1] > high_12h[i]:
            bullish_ob[i] = True
        # Bearish OB: candle i is bullish (close > open), candle i+1 is bearish and closes < low[i]
        if close_12h[i] > open_12h[i] and close_12h[i+1] < open_12h[i+1] and close_12h[i+1] < low_12h[i]:
            bearish_ob[i] = True
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current 6h volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Align all indicators to 6h timeframe
    bullish_ob_aligned = align_htf_to_ltf(prices, df_12h, bullish_ob.astype(float))
    bearish_ob_aligned = align_htf_to_ltf(prices, df_12h, bearish_ob.astype(float))
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_12h, volume_confirm.astype(float))  # align using 12h as reference
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need EMA50 (50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bullish_ob_aligned[i]) or np.isnan(bearish_ob_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        bull_ob = bullish_ob_aligned[i] > 0.5
        bear_ob = bearish_ob_aligned[i] > 0.5
        ema50 = ema50_12h_aligned[i]
        vol_conf = volume_confirm_aligned[i] > 0.5
        
        if position == 0:
            # Long when price breaks above bullish OB with uptrend and volume
            if close_val > ema50 and vol_conf:
                # Check if we're above any bullish OB level (use recent OB)
                # Simplified: if price > recent bullish OB area
                if bull_ob:  # current candle is bullish OB
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            # Short when price breaks below bearish OB with downtrend and volume
            elif close_val < ema50 and vol_conf:
                if bear_ob:  # current candle is bearish OB
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit: close below 12h EMA50 or bearish OB
            if close_val < ema50 or bear_ob:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: close above 12h EMA50 or bullish OB
            if close_val > ema50 or bull_ob:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_OrderBlock_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0