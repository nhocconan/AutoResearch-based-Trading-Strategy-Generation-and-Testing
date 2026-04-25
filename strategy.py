#!/usr/bin/env python3
"""
12h Camarilla R1/S1 Breakout with 1d EMA34 Trend and Volume Spike
Hypothesis: Camarilla pivot levels act as intraday support/resistance. 
Breakouts above R1 (bullish) or below S1 (bearish) with 1d EMA34 trend alignment 
and volume spikes capture strong moves. Uses 12h timeframe to avoid overtrading 
and fee drag, targeting 50-150 total trades over 4 years (12-37/year).
Works in both bull/bear via trend filter and volume confirmation to avoid false breakouts.
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
    
    # Get 1d data for Camarilla pivots and EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # Typical Price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Camarilla width = (High - Low) * 1.1 / 12
    camarilla_width = (df_1d['high'] - df_1d['low']) * 1.1 / 12
    # R1 = Close + (width * 1.1)
    r1 = df_1d['close'] + camarilla_width * 1.1
    # S1 = Close - (width * 1.1)
    s1 = df_1d['close'] - camarilla_width * 1.1
    
    # Align Camarilla levels to 12h timeframe (1-bar delay for completed 1d bar)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume MA for 12h volume confirmation
    vol_ma_20_12h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_12h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34 and volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_20_12h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        vol_ma_12h = vol_ma_20_12h[i]
        
        # Volume confirmation: current 12h volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma_12h
        
        if position == 0:
            # Look for entry signals
            # Long: Break above R1 AND price > EMA34 (uptrend) AND volume confirmation
            long_entry = (curr_high > r1_level and 
                         curr_close > ema_trend and volume_confirm)
            # Short: Break below S1 AND price < EMA34 (downtrend) AND volume confirmation
            short_entry = (curr_low < s1_level and 
                          curr_close < ema_trend and volume_confirm)
            
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
            # Exit: Close below S1 (reversal) OR price falls below EMA34
            if (curr_close < s1_level or curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Close above R1 (reversal) OR price rises above EMA34
            if (curr_close > r1_level or curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0