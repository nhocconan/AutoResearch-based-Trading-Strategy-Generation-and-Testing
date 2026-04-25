#!/usr/bin/env python3
"""
12h Camarilla H3/L3 Breakout with 1d EMA34 Trend Filter and Volume Spike
Hypothesis: Camarilla pivot levels (H3/L3) from 1-day timeframe act as strong support/resistance.
Breaks above H3 or below L3 with volume confirmation (>2.0x 20-bar vol MA) and 1d EMA34 trend filter
provide high-probability entries. This strategy works in bull markets via longs above H3 in uptrend
(price > 1d EMA34) and in bear markets via shorts below L3 in downtrend (price < 1d EMA34).
The 1d EMA34 filter reduces whipsaws in choppy markets and improves generalization to bear markets (2025+).
Target: 50-150 total trades over 4 years = 12-37/year. Size: 0.25.
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
    
    # Get 1d data for Camarilla pivot and EMA (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # We focus on H3 and L3 for breakout
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla H3 and L3
    camarilla_h3 = prev_close + prev_range * 1.1 / 4
    camarilla_l3 = prev_close - prev_range * 1.1 / 4
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 12h timeframe
    camarilla_h3_1d = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_1d = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    ema_34_1d = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 20-period volume MA for volume spike (12h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 1d EMA34 calculation and volume MA
    start_idx = max(34, 20)  # 34 for 1d EMA, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_1d[i]) or 
            np.isnan(camarilla_l3_1d[i]) or 
            np.isnan(ema_34_1d[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        camarilla_h3_val = camarilla_h3_1d[i]
        camarilla_l3_val = camarilla_l3_1d[i]
        ema_34_val = ema_34_1d[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Breakout conditions
        breakout_above_h3 = curr_high > camarilla_h3_val
        breakout_below_l3 = curr_low < camarilla_l3_val
        
        # Trend filter: price relative to 1d EMA34
        price_above_ema = curr_close > ema_34_val
        price_below_ema = curr_close < ema_34_val
        
        if position == 0:
            # Long: breakout above H3 + price above 1d EMA34 + volume confirmation
            long_signal = breakout_above_h3 and price_above_ema and volume_confirm
            # Short: breakout below L3 + price below 1d EMA34 + volume confirmation
            short_signal = breakout_below_l3 and price_below_ema and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below camarilla L3 OR price crosses below 1d EMA34
            if curr_low < camarilla_l3_val or curr_close < ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above camarilla H3 OR price crosses above 1d EMA34
            if curr_high > camarilla_h3_val or curr_close > ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0