#!/usr/bin/env python3
"""
6h Williams Alligator + 1d EMA50 Trend + Volume Spike
Hypothesis: The Williams Alligator (JAW/TEETH/LIPS) identifies trending vs ranging markets.
In strong trends (Alligator aligned and mouth open), we trade breakouts in the direction of the 1d EMA50 trend.
Volume confirmation avoids false breakouts. Works in bull/bear by following the higher timeframe trend.
Discrete sizing (0.25) targets ~50-120 trades over 4 years.
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
    
    # Get daily data for EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR for stop loss (using 21 periods)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Williams Alligator on 6h data: SMAs of median price
    # Jaw: 13-period SMMA, Teeth: 8-period, Lips: 5-period
    # SMMA = smoothed moving average (similar to Wilder's smoothing)
    median_price = (high + low) / 2
    
    # Smoothed moving average (SMMA) - similar to RMA/Wilder's
    def smma(source, period):
        result = np.full_like(source, np.nan, dtype=np.float64)
        if period < 1 or len(source) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for Alligator (13) and ATR (21)
    start_idx = max(21, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_50_aligned[i]
        atr_value = atr[i]
        
        # Alligator conditions:
        # Trending: Alligator lines are separated and ordered
        # Mouth open: (JAW - LIPS) > 0 for uptrend, (LIPS - JAW) > 0 for downtrend
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Alligator aligned for uptrend: Lips > Teeth > Jaw
        alligator_long_align = lips_val > teeth_val and teeth_val > jaw_val
        # Alligator aligned for downtrend: Jaw > Teeth > Lips
        alligator_short_align = jaw_val > teeth_val and teeth_val > lips_val
        # Mouth open (separation) - avoid choppy markets
        mouth_open = abs(jaw_val - lips_val) > (atr_value * 0.5)
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = np.mean(volume[:i]) if i > 0 else volume[i]
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout logic: in trending markets, trade in direction of Alligator alignment
        # Long: Alligator bullish aligned + price above Teeth (or Jaw) + EMA50 up + volume
        # Short: Alligator bearish aligned + price below Teeth (or Jaw) + EMA50 down + volume
        if position == 0:
            long_condition = (alligator_long_align and mouth_open and 
                            curr_close > teeth_val and 
                            curr_close > ema_trend and 
                            volume_spike)
            short_condition = (alligator_short_align and mouth_open and 
                             curr_close < teeth_val and 
                             curr_close < ema_trend and 
                             volume_spike)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            elif short_condition:
                signals[i] = -0.25
                position = -1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
        elif position == 1:
            # Long position management
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit: price closes below Lips (Alligator wake-up call) or 3*ATR trailing stop
            if curr_close < lips_val or curr_close < highest_since_entry - 3.0 * atr_value:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit: price closes above Lips or 3*ATR trailing stop
            if curr_close > lips_val or curr_close > lowest_since_entry + 3.0 * atr_value:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0