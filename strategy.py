#!/usr/bin/env python3
"""
1h Camarilla H3/L3 Breakout with 4h EMA34 Trend and Volume Spike + Session Filter
Hypothesis: Camarilla H3/L3 levels on 4h act as key support/resistance. Breakouts with
volume confirmation and aligned 4h EMA34 trend capture continuation moves. Session filter
(08-20 UTC) reduces noise. Designed for 1h timeframe with 15-37 trades/year to minimize
fee drag while working in both bull/bear markets via EMA trend filter.
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
    
    # Get 4h data for EMA34 trend and Camarilla levels (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 4h close for trend
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 4h OHLC for Camarilla calculation (based on previous 4h bar)
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar
    # H3 = close + (high - low) * 1.1/4
    # L3 = close - (high - low) * 1.1/4
    camarilla_h3_4h = c_4h + (h_4h - l_4h) * 1.1 / 4
    camarilla_l3_4h = c_4h - (h_4h - l_4h) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (each 4h level lasts for 4 bars)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3_4h)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3_4h)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA, ATR, volume MA, and Camarilla
    start_idx = max(50, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_4h_aligned[i]
        h3_level = camarilla_h3_aligned[i]
        l3_level = camarilla_l3_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 level AND volume spike AND price > 4h EMA34 (uptrend)
            long_entry = (curr_close > h3_level) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below L3 level AND volume spike AND price < 4h EMA34 (downtrend)
            short_entry = (curr_close < l3_level) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below L3 level (reversal) OR price crosses below EMA (trend change)
            if (curr_close < l3_level) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: price crosses above H3 level (reversal) OR price crosses above EMA (trend change)
            if (curr_close > h3_level) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA34_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0