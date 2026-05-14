#!/usr/bin/env python3
"""
4h Camarilla H4/L4 Breakout with 1d EMA50 Trend Filter and Volume Spike Confirmation - Reduced Frequency
Hypothesis: Camarilla pivot levels H4/L4 act as stronger breakout levels than H3/L3, requiring more significant momentum. Combined with 1d EMA50 trend filter and volume spike (>2.0x 20-bar vol MA) to capture strong momentum moves while minimizing false breakouts. Works in bull markets via upside breakouts above H4 and in bear markets via downside breakdowns below L4. Targeting 20-50 trades per year to avoid fee drag.
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
    
    # Get 1d data for EMA50 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 51:  # Need 50 for EMA + 1 for shift
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous day (using 1d data)
    # Camarilla: H4 = C + (H-L)*1.1/2, L4 = C - (H-L)*1.1/2
    # where C = close, H = high, L = low of previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    camarilla_h4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_l4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 20-period volume MA for volume spike confirmation (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA50, Camarilla, and volume MA
    start_idx = max(51, 20)  # 51 for EMA50 (50 + 1 for shift), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50_val = ema_50_1d_aligned[i]
        h4_val = camarilla_h4_aligned[i]
        l4_val = camarilla_l4_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = curr_close > ema_50_val
        price_below_ema = curr_close < ema_50_val
        
        if position == 0:
            # Long: break above H4 + price above 1d EMA50 + volume confirmation
            long_signal = (curr_high > h4_val) and price_above_ema and volume_confirm
            # Short: break below L4 + price below 1d EMA50 + volume confirmation
            short_signal = (curr_low < l4_val) and price_below_ema and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below H4 OR price crosses below 1d EMA50
            if (curr_close < h4_val) or (curr_close < ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above L4 OR price crosses above 1d EMA50
            if (curr_close > l4_val) or (curr_close > ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H4L4_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0