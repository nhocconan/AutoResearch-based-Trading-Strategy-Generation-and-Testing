#!/usr/bin/env python3
"""
4h Camarilla H3L3 Breakout with 12h EMA Trend Filter and Volume Spike
Hypothesis: Camarilla H3 (resistance) and L3 (support) levels act as key intraday pivot points. 
Breakout above H3 with volume confirmation and 12h EMA50 uptrend triggers longs; 
breakdown below L3 with volume and 12h EMA50 downtrend triggers shorts. 
Using discrete sizing (0.0, ±0.25) to minimize fee churn. Target: 20-50 trades/year on 4h.
Works in bull markets (breakouts above H3 in uptrend) and bear markets (breakdowns below L3 in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period Camarilla levels (using prior 4h bar's HLC)
    # We need to shift by 1 to avoid look-ahead: use previous bar's high/low/close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # fill first value
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla levels: H3, L3, H4, L4
    # H3 = Close + 1.1*(High-Low)/2
    # L3 = Close - 1.1*(High-Low)/2
    # H4 = Close + 1.1*(High-Low)
    # L4 = Close - 1.1*(High-Low)
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_50_12h_aligned[i]
        vol_spike = volume_spike[i]
        h3_level = camarilla_h3[i]
        l3_level = camarilla_l3[i]
        h4_level = camarilla_h4[i]
        l4_level = camarilla_l4[i]
        
        if position == 0:
            # Look for entry signals
            # Long: break above H3 with volume spike AND price > 12h EMA50 (uptrend)
            long_entry = (curr_high > h3_level) and vol_spike and (curr_close > ema_trend)
            # Short: break below L3 with volume spike AND price < 12h EMA50 (downtrend)
            short_entry = (curr_low < l3_level) and vol_spike and (curr_close < ema_trend)
            
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
            # Exit: price falls below L3 OR price < 12h EMA50 (trend change)
            if (curr_low < l3_level) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above H3 OR price > 12h EMA50 (trend change)
            if (curr_high > h3_level) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_VolumeSpike_12hEMA50_Trend"
timeframe = "4h"
leverage = 1.0