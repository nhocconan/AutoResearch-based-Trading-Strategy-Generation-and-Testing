#!/usr/bin/env python3
"""
12h Williams Alligator + 1d EMA50 Trend + Volume Spike
Hypothesis: Williams Alligator (jaw/teeth/lips) identifies trend phases. Trade only when all three lines are aligned (bullish/bearish) with price outside the Alligator mouth. 1d EMA50 filter ensures alignment with higher-timeframe trend. Volume spike confirms participation. Works in bull via buying when Alligator awakens uptrend, bear via selling when Alligator awakens downtrend. Uses discrete position sizing (0.25) to control drawdown. Target: 12-37 trades/year on 12h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_alligator

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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Pre-compute 20-period volume MA for volume spike detection
    vol_ma_20 = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    # Calculate Williams Alligator on 12h data (jaw=13, teeth=8, lips=5)
    # Using SMMA (smoothed moving average) as per original Alligator
    jaw_period, teeth_period, lips_period = 13, 8, 5
    jaw_shift, teeth_shift, lips_shift = 8, 5, 3
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA*(period-1) + CURRENT_PRICE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, jaw_period)
    teeth = smma(close, teeth_period)
    lips = smma(close, lips_period)
    
    # Apply shifts (Alligator lines are shifted into the future)
    jaw_aligned = np.roll(jaw, jaw_shift)
    teeth_aligned = np.roll(teeth, teeth_shift)
    lips_aligned = np.roll(lips, lips_shift)
    
    # Invalidate the shifted portions
    jaw_aligned[:jaw_shift] = np.nan
    teeth_aligned[:teeth_shift] = np.nan
    lips_aligned[:lips_shift] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Alligator to stabilize
    start_idx = max(jaw_shift, teeth_shift, lips_shift) + 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr[i]) or
            np.isnan(jaw_aligned[i]) or
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50 = ema_50_1d_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Alligator alignment: bullish when lips > teeth > jaw, bearish when lips < teeth < jaw
        bullish_align = lips_val > teeth_val > jaw_val
        bearish_align = lips_val < teeth_val < jaw_val
        
        # Price outside Alligator mouth: above highest line for long, below lowest for short
        alligator_high = max(jaw_val, teeth_val, lips_val)
        alligator_low = min(jaw_val, teeth_val, lips_val)
        price_above_mouth = curr_close > alligator_high
        price_below_mouth = curr_close < alligator_low
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long: bullish Alligator alignment AND price above mouth AND above 1d EMA50 AND volume spike
            long_condition = bullish_align and price_above_mouth and curr_close > ema_50 and volume_spike
            # Short: bearish Alligator alignment AND price below mouth AND below 1d EMA50 AND volume spike
            short_condition = bearish_align and price_below_mouth and curr_close < ema_50 and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or Alligator turns bearish or price falls below 1d EMA50
            if curr_close <= entry_price - 2.0 * atr_val or not bullish_align or curr_close < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or Alligator turns bullish or price rises above 1d EMA50
            if curr_close >= entry_price + 2.0 * atr_val or not bearish_align or curr_close > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0