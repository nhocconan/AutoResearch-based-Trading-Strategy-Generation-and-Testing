#!/usr/bin/env python3
"""
1d Camarilla Pivot H3/L3 Breakout + 1w EMA34 Trend + Volume Spike
Hypothesis: On daily timeframe, Camarilla H3 (resistance) and L3 (support) levels act as 
key intraday pivot points. Breakouts above H3 or below L3 with 1-week EMA34 trend alignment 
and volume confirmation capture strong momentum moves. Works in bull/bear via higher timeframe 
trend filter. Target: 10-25 trades/year on 1d to minimize fee drag.
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
    
    # Load 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need at least 2 days for Camarilla calculation
    start_idx = 2
    
    for i in range(start_idx, n):
        # Skip if trend data not ready
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels from previous day (HLC of i-1)
        if i < 1:
            signals[i] = 0.0
            continue
            
        high_prev = high[i-1]
        low_prev = low[i-1]
        close_prev = close[i-1]
        
        # Camarilla pivot levels
        pivot = (high_prev + low_prev + close_prev) / 3.0
        range_prev = high_prev - low_prev
        
        # H3 and L3 levels
        H3 = high_prev + range_prev * 1.1 / 4.0
        L3 = low_prev - range_prev * 1.1 / 4.0
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: price relative to 1w EMA34
        bullish_bias = curr_close > ema_1w_aligned[i]
        bearish_bias = curr_close < ema_1w_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_spike = curr_volume > (vol_ma * 2.0)
        else:
            volume_spike = False
        
        if position == 0:
            # Look for entry signals - require: Camarilla breakout + trend + volume
            # Long: price breaks above H3 AND bullish bias AND volume spike
            long_entry = curr_close > H3 and bullish_bias and volume_spike
            # Short: price breaks below L3 AND bearish bias AND volume spike
            short_entry = curr_close < L3 and bearish_bias and volume_spike
            
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
            # Exit: price crosses below pivot (mean reversion) OR loss of bullish bias
            if (curr_close < pivot) or (curr_close < ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above pivot (mean reversion) OR loss of bearish bias
            if (curr_close > pivot) or (curr_close > ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0