#!/usr/bin/env python3
"""
6h Camarilla H4L4 Breakout with Weekly EMA50 Trend Filter and Volume Spike Confirmation
Hypothesis: Camarilla pivot levels (H4/L4) act as strong intraday support/resistance on 6h timeframe.
Breakouts above H4 or below L4 with volume confirmation (>2.0x 20-bar vol MA) and aligned with 
weekly EMA50 trend capture strong momentum moves. Works in both bull (buy breakouts in uptrend) 
and bear (sell breakdowns in downtrend) markets by using weekly trend filter. Target: 12-30 trades/year.
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
    
    # Get weekly data for EMA50 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 51:  # Need 50 for EMA + 1 for shift
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'])
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period volume MA for volume spike confirmation (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for weekly EMA50 and volume MA
    start_idx = max(51, 20)  # 51 for weekly EMA50 (50 + 1 for shift), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Calculate Camarilla pivot levels for 6h bar using previous bar's OHLC
        # H4 = Close + 1.5 * (High - Low)
        # L4 = Close - 1.5 * (High - Low)
        # We use previous bar's OHLC to avoid look-ahead
        if i > 0:
            prev_close = close[i-1]
            prev_high = high[i-1]
            prev_low = low[i-1]
            camarilla_range = prev_high - prev_low
            h4 = prev_close + 1.5 * camarilla_range
            l4 = prev_close - 1.5 * camarilla_range
        else:
            # Not enough data for first bar
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below weekly EMA50
        price_above_ema = curr_close > ema_50_val
        price_below_ema = curr_close < ema_50_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long breakout: price breaks above H4 with volume confirmation and uptrend
            long_signal = (curr_high > h4) and price_above_ema and volume_confirm
            # Short breakdown: price breaks below L4 with volume confirmation and downtrend
            short_signal = (curr_low < l4) and price_below_ema and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below weekly EMA50 OR reversal signal (close below H4)
            if (curr_close < ema_50_val) or (curr_close < h4):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above weekly EMA50 OR reversal signal (close above L4)
            if (curr_close > ema_50_val) or (curr_close > l4):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H4L4_Breakout_WeeklyEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0