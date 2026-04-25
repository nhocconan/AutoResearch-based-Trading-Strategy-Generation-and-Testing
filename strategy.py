#!/usr/bin/env python3
"""
12h Camarilla H3/L3 Breakout with 1d EMA34 Trend Filter and Volume Spike Confirmation
Hypothesis: Camarilla H3/L3 levels from daily pivot act as key support/resistance. A break above H3 with
1d EMA34 uptrend and volume spike (>2x 20-bar vol MA) signals bullish momentum. Break below L3 with
1d EMA34 downtrend and volume spike signals bearish momentum. Uses discrete position sizing (0.25) to limit fee drag.
Designed for 12h timeframe to target 50-150 trades over 4 years. Works in bull markets via upside breakouts
and in bear markets via downside breakdowns.
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
    
    # Get 1d data for EMA34 trend and Camarilla levels (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day (using 1d data)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: H3/L3 = C + (H-L)*1.1/2 and C - (H-L)*1.1/2
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (no extra delay - levels known at bar open)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 20-period volume MA for volume spike confirmation (12h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34 calculation, volume MA, and Camarilla alignment
    start_idx = max(34, 20)  # 34 for EMA34, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34_val = ema_34_1d_aligned[i]
        h3_level = camarilla_h3_aligned[i]
        l3_level = camarilla_l3_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average (tighter for fewer trades)
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Breakout conditions
        breakout_above_h3 = curr_close > h3_level
        breakout_below_l3 = curr_close < l3_level
        
        # Trend filter: price relative to 1d EMA34
        price_above_ema = curr_close > ema_34_val
        price_below_ema = curr_close < ema_34_val
        
        if position == 0:
            # Long: break above H3 + price above 1d EMA34 + volume confirmation
            long_signal = breakout_above_h3 and price_above_ema and volume_confirm
            # Short: break below L3 + price below 1d EMA34 + volume confirmation
            short_signal = breakout_below_l3 and price_below_ema and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below H3 level OR price crosses below 1d EMA34
            if curr_close < h3_level or curr_close < ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above L3 level OR price crosses above 1d EMA34
            if curr_close > l3_level or curr_close > ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0