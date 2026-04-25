#!/usr/bin/env python3
"""
1d Camarilla Pivot H3/L3 Breakout with 1w EMA34 Trend Filter and Volume Spike
Hypothesis: Camarilla pivot levels (H3/L3) act as strong intraday support/resistance. A breakout above H3 or below L3 with volume confirmation indicates institutional participation. Using 1w EMA34 as higher-timeframe trend filter ensures alignment with weekly trend, reducing false signals in choppy markets. Works in bull markets (breakouts above H3) and bear markets (breakdowns below L3) by requiring trend alignment. Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 15-25 trades/year on 1d.
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
    
    # Get 1w data for EMA trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Need previous day's high, low, close for Camarilla calculation
        if i == 0:
            continue
            
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Calculate Camarilla pivot levels for today (based on previous day)
        range_val = prev_high - prev_low
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        # Camarilla levels
        H3 = prev_close + range_val * 1.1 / 4
        L3 = prev_close - range_val * 1.1 / 4
        H4 = prev_close + range_val * 1.1 / 2
        L4 = prev_close - range_val * 1.1 / 2
        
        curr_close = close[i]
        curr_volume = volume[i]
        avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        ema_trend = ema_34_1w_aligned[i]
        
        # Volume spike: current volume > 1.5 * 20-day average
        volume_spike = curr_volume > 1.5 * avg_volume if not np.isnan(avg_volume) else False
        
        if position == 0:
            # Look for entry signals
            # Long: break above H3 with volume spike and price > weekly EMA34 (uptrend)
            long_entry = (curr_close > H3) and volume_spike and (curr_close > ema_trend)
            # Short: break below L3 with volume spike and price < weekly EMA34 (downtrend)
            short_entry = (curr_close < L3) and volume_spike and (curr_close < ema_trend)
            
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
            # Exit: price falls back below H3 (failed breakout) or weekly trend turns down
            if (curr_close < H3) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises back above L3 (failed breakdown) or weekly trend turns up
            if (curr_close > L3) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_VolumeSpike_1wEMA34_Trend"
timeframe = "1d"
leverage = 1.0