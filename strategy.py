#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + 1w EMA34 trend filter + volume confirmation.
# Uses Williams Alligator (Jaw/Teeth/Lips) on 1d for structure, with 1w EMA34 for primary trend.
# Enter long when Lips > Teeth > Jaw (bullish alignment) and price > Lips, with volume > 1.5x 20-bar average.
# Enter short when Lips < Teeth < Jaw (bearish alignment) and price < Lips, with volume confirmation.
# ATR(14) trailing stop at 3.0x for risk management. Discrete position sizing at ±0.25.
# Target: 30-100 total trades over 4 years (7-25/year). Works in both bull and bear markets
# by requiring HTF trend alignment to avoid counter-trend whipsaws and using Alligator's
# natural filtering to reduce false signals during ranging periods.

name = "1d_WilliamsAlligator_1wEMA34_VolumeConfirm_ATRStop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d data
    # Jaw: Blue line (13-period SMMA, shifted 8 bars)
    # Teeth: Red line (8-period SMMA, shifted 5 bars)
    # Lips: Green line (5-period SMMA, shifted 3 bars)
    def smma(values, period):
        """Smoothed Moving Average"""
        result = np.full_like(values, np.nan, dtype=np.float64)
        if len(values) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA*(period-1) + CURRENT_VALUE) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    close_1d = df_1d['close'].values
    jaw = smma(close_1d, 13)  # 13-period
    teeth = smma(close_1d, 8)  # 8-period
    lips = smma(close_1d, 5)   # 5-period
    
    # Shift the lines as per Alligator definition
    jaw = np.roll(jaw, 8)   # shifted 8 bars
    teeth = np.roll(teeth, 5) # shifted 5 bars
    lips = np.roll(lips, 3)   # shifted 3 bars
    
    # Align Alligator lines to primary timeframe (1d -> 1d: identity but using helper for consistency)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Load 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # ATR(14) for volatility and stoploss
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(13, 8, 5, 34, atr_period, 20) + 1  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(jaw_aligned[i]) or
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_jaw = jaw_aligned[i]
        curr_teeth = teeth_aligned[i]
        curr_lips = lips_aligned[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_align = curr_lips > curr_teeth > curr_jaw
        # Bearish alignment: Lips < Teeth < Jaw
        bearish_align = curr_lips < curr_teeth < curr_jaw
        
        if position == 0:  # Flat - look for new entries
            # Long: bullish alignment, price > Lips, above 1w EMA34, volume confirmation
            if (bullish_align and 
                curr_close > curr_lips and 
                curr_close > curr_ema_34_1w and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            # Short: bearish alignment, price < Lips, below 1w EMA34, volume confirmation
            elif (bearish_align and 
                  curr_close < curr_lips and 
                  curr_close < curr_ema_34_1w and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR trailing stop: exit if price drops 3.0*ATR from highest point
            if curr_close < highest_since_entry - (3.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: exit if price rises 3.0*ATR from lowest point
            if curr_close > lowest_since_entry + (3.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals