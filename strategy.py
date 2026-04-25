#!/usr/bin/env python3
"""
6h Camarilla H3/L3 Breakout with 12h EMA50 Trend Filter and Volume Spike
Hypothesis: Camarilla H3/L3 levels act as significant intraday support/resistance. 
Breakouts above H3 or below L3 with 12h EMA50 trend alignment and volume confirmation 
capture strong momentum moves. Designed for 6h timeframe to target 12-37 trades/year 
(50-150 over 4 years) by requiring confluence of Camarilla breakout, 12h EMA50 trend, 
and volume confirmation, reducing overtrading and fee drag while working in both bull 
and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Camarilla calculation (uses previous day)
    start_idx = 24  # 4 * 6h bars = 1 day minimum for Camarilla
    
    for i in range(start_idx, n):
        # Calculate Camarilla levels using previous 12h bar's OHLC (6h bars ago)
        # Since we're on 6h timeframe, previous day = 4 bars ago
        prev_idx = i - 4
        if prev_idx < 0:
            signals[i] = 0.0
            continue
            
        # Get previous day's OHLC (4 bars ago on 6h chart)
        prev_high = high[prev_idx]
        prev_low = low[prev_idx]
        prev_close = close[prev_idx]
        
        # Camarilla levels calculation
        range_val = prev_high - prev_low
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        # Camarilla H3 and L3 levels
        h3 = prev_close + (range_val * 1.1 / 4)
        l3 = prev_close - (range_val * 1.1 / 4)
        h4 = prev_close + (range_val * 1.1 / 2)
        l4 = prev_close - (range_val * 1.1 / 2)
        
        # Skip if any data not ready
        if np.isnan(ema_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 12h EMA50
        bullish_bias = curr_close > ema_12h_aligned[i]
        bearish_bias = curr_close < ema_12h_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Camarilla breakout + trend + volume
            # Volume confirmation: current volume > 1.5 * 24-period average (4 days of 6h)
            if i >= 24:
                vol_ma = np.mean(volume[i-24:i]) if i >= 24 else volume[i]
                volume_spike = volume[i] > (vol_ma * 1.5)
            else:
                volume_spike = False
            
            # Long: price breaks above H3 AND bullish bias AND volume spike
            long_entry = (curr_high > h3) and bullish_bias and volume_spike
            # Short: price breaks below L3 AND bearish bias AND volume spike
            short_entry = (curr_low < l3) and bearish_bias and volume_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below L3 (mean reversion) OR loss of bullish bias
            if (curr_low < l3) or (curr_close < ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above H3 (mean reversion) OR loss of bearish bias
            if (curr_high > h3) or (curr_close > ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0