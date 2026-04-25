#!/usr/bin/env python3
"""
4h Williams Alligator + 1d EMA34 Trend + Volume Confirmation
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) on 4h identifies trend absence (all lines intertwined = ranging market).
When Alligator "awakens" (lines diverge) in direction of 1d EMA34 trend with volume confirmation,
it captures the start of sustained moves. Works in bull via long awakenings, bear via short awakenings.
Target: 20-50 trades/year on 4h.
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
    
    # Get 4h data for Williams Alligator
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 4h: SMAs of median price
    # Jaw (Blue): 13-period SMMA shifted 8 bars
    # Teeth (Red): 8-period SMMA shifted 5 bars  
    # Lips (Green): 5-period SMMA shifted 3 bars
    median_price_4h = (df_4h['high'].values + df_4h['low'].values) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full(len(arr), np.nan)
        result = np.full(len(arr), np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (Prev SMMA * (Period-1) + Current Price) / Period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_4h = smma(median_price_4h, 13)
    teeth_4h = smma(median_price_4h, 8)
    lips_4h = smma(median_price_4h, 5)
    
    # Shift as per Alligator definition
    jaw_4h = np.roll(jaw_4h, 8)
    teeth_4h = np.roll(teeth_4h, 5)
    lips_4h = np.roll(lips_4h, 3)
    
    # Align with 1-bar delay (wait for 4h bar close)
    jaw_4h_aligned = align_htf_to_ltf(prices, df_4h, jaw_4h)
    teeth_4h_aligned = align_htf_to_ltf(prices, df_4h, teeth_4h)
    lips_4h_aligned = align_htf_to_ltf(prices, df_4h, lips_4h)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_4h_aligned[i]) or 
            np.isnan(teeth_4h_aligned[i]) or 
            np.isnan(lips_4h_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        jaw = jaw_4h_aligned[i]
        teeth = teeth_4h_aligned[i]
        lips = lips_4h_aligned[i]
        ema_34 = ema_34_1d_aligned[i]
        atr_val = atr[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Alligator conditions: lines must be separated (not intertwined)
        # Bullish: Lips > Teeth > Jaw (green above red above blue)
        # Bearish: Lips < Teeth < Jaw (green below red below blue)
        bullish_alligator = lips > teeth and teeth > jaw
        bearish_alligator = lips < teeth and teeth < jaw
        
        # Trend filter from 1d EMA34
        uptrend = curr_close > ema_34
        downtrend = curr_close < ema_34
        
        if position == 0:
            # Long: Alligator bullish AND uptrend AND volume spike
            long_condition = bullish_alligator and uptrend and volume_spike
            # Short: Alligator bearish AND downtrend AND volume spike
            short_condition = bearish_alligator and downtrend and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or Alligator turns bearish
            if curr_close <= entry_price - 2.0 * atr_val or not bullish_alligator:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or Alligator turns bullish
            if curr_close >= entry_price + 2.0 * atr_val or not bearish_alligator:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA34_Trend_VolumeConfirmation_v1"
timeframe = "4h"
leverage = 1.0