#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with 1d EMA50 trend filter and volume confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) from 6h for trend direction and alignment
# Elder Ray (Bull/Bear Power) from 6h for momentum confirmation
# 1d EMA50 as higher timeframe trend filter to avoid counter-trend trades
# Volume spike (>2.0x 24-period average) ensures institutional participation
# Designed for 6h timeframe to capture medium-term swings with controlled frequency (target: 50-150 trades over 4 years)
# Works in both bull and bear markets by requiring trend alignment across multiple timeframes

name = "6h_WilliamsAlligator_ElderRay_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator components from 6h data
    # Jaw: 13-period SMMA (smoothed moving average) of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2
    
    # Smoothed Moving Average (SMMA) calculation
    def smma(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CURRENT_VALUE) / PERIOD
        for i in range(period, len(values)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Shift the lines as per Alligator specification
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Calculate Elder Ray components
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 24-period average volume for confirmation
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 24, 13, 21)  # EMA50_1d, volume MA, EMA13, and Alligator warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_24[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Alligator lines converge (teeth crosses below lips) OR bear power turns negative
            if curr_teeth < curr_lips or curr_bear_power > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator lines converge (teeth crosses above lips) OR bull power turns positive
            if curr_teeth > curr_lips or curr_bull_power < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 24-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Alligator alignment: Lips > Teeth > Jaw for uptrend, Lips < Teeth < Jaw for downtrend
            alligator_long = curr_lips > curr_teeth > curr_jaw
            alligator_short = curr_lips < curr_teeth < curr_jaw
            
            # Long entry: Alligator aligned up, Bull Power positive, above 1d EMA50, volume confirmation
            if (vol_confirm and alligator_long and curr_bull_power > 0 and 
                curr_close > curr_ema50_1d):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: Alligator aligned down, Bear Power negative, below 1d EMA50, volume confirmation
            elif (vol_confirm and alligator_short and curr_bear_power < 0 and 
                  curr_close < curr_ema50_1d):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals