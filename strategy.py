#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams Alligator with volume confirmation and 4h trend filter
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trending vs ranging markets.
# When all three lines are aligned (Jaw > Teeth > Lips for uptrend, reverse for downtrend),
# it indicates a strong trend. Breakouts with volume confirmation capture momentum moves.
# Designed for low trade frequency (<25/year) to minimize fee drag in both bull and bear markets.
# Uses 1d timeframe for Alligator calculation to reduce noise and false signals.

name = "4h_Williams_Alligator_1dTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # need enough for Jaw (13+8+5+8 smoothing)
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d timeframe
    # Jaw (Blue): 13-period SMMA, shifted 8 bars forward
    # Teeth (Red): 8-period SMMA, shifted 5 bars forward  
    # Lips (Green): 5-period SMMA, shifted 3 bars forward
    close_1d = df_1d['close'].values
    
    # Smoothed Moving Average (SMMA) - same as RMA/Wilder's MA
    def smma(data, period):
        if len(data) < period:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (Prev SMMA * (Period-1) + Current Close) / Period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw_raw = smma(close_1d, 13)
    teeth_raw = smma(close_1d, 8)
    lips_raw = smma(close_1d, 5)
    
    # Apply shifts (Jaw shifted 8, Teeth shifted 5, Lips shifted 3)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Align Alligator lines to 4h timeframe (wait for completed 1d bar)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate 4h EMA(34) for trend filter
    close_s = pd.Series(close)
    ema_34 = close_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 34  # warmup for EMA(34)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (2.0 * vol_ma_20)
        
        curr_close = close[i]
        curr_ema = ema_34[i]
        curr_atr = atr[i]
        curr_jaw = jaw_aligned[i]
        curr_teeth = teeth_aligned[i]
        curr_lips = lips_aligned[i]
        
        # Check for valid Alligator values (not NaN)
        if np.isnan(curr_jaw) or np.isnan(curr_teeth) or np.isnan(curr_lips):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and Alligator alignment
            if volume_spike:
                # Bullish entry: Alligator aligned for uptrend (Jaw > Teeth > Lips) and price above EMA
                if curr_jaw > curr_teeth > curr_lips and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Alligator aligned for downtrend (Jaw < Teeth < Lips) and price below EMA
                elif curr_jaw < curr_teeth < curr_lips and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR Alligator loses alignment
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif not (curr_jaw > curr_teeth > curr_lips):  # Alligator alignment broken
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x ATR above entry
            elif curr_close >= entry_price + 1.5 * curr_atr:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR Alligator loses alignment
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif not (curr_jaw < curr_teeth < curr_lips):  # Alligator alignment broken
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x ATR below entry
            elif curr_close <= entry_price - 1.5 * curr_atr:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals