#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation.
# Long when price is above Alligator lips (green line), price above 1d EMA34, and volume > 1.5x 20-period average.
# Short when price is below Alligator lips, price below 1d EMA34, and volume > 1.5x 20-period average.
# Exit when price crosses Alligator teeth (red line) or trend fails.
# Williams Alligator identifies trend presence and direction; EMA34 confirms higher timeframe trend; volume confirms strength.
# Designed to capture strong trends while avoiding false signals in ranging markets.
# Target: 12-37 trades/year to stay within profitable range.

name = "12h_Williams_Alligator_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Williams Alligator lines (SMMA: Smoothed Moving Average)
    def smma(series, period):
        result = np.full_like(series, np.nan, dtype=float)
        if len(series) < period:
            return result
        # First value is simple moving average
        result[period-1] = np.mean(series[:period])
        # Subsequent values: (prev * (period-1) + current) / period
        for i in range(period, len(series)):
            result[i] = (result[i-1] * (period-1) + series[i]) / period
        return result
    
    # Alligator lines: Jaw (blue, 13-period, 8 bars future), Teeth (red, 8-period, 5 bars future), Lips (green, 5-period, 3 bars future)
    jaw = smma(close_12h, 13)
    teeth = smma(close_12h, 8)
    lips = smma(close_12h, 5)
    
    # Shift jaws forward: Jaw 8 bars, Teeth 5 bars, Lips 3 bars
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Fill shifted values with NaN for invalid positions
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Calculate 12h 20-period average volume for volume filter
    vol_ma_20 = smma(volume_12h, 20)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Calculate EMA34 slope for trend direction (rising/falling)
    ema_34_prev = np.roll(ema_34, 1)
    ema_34_prev[0] = ema_34[0]
    ema_34_rising = ema_34 > ema_34_prev
    ema_34_falling = ema_34 < ema_34_prev
    
    # Align all indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_shifted)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_falling)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or \
           np.isnan(vol_ma_20_aligned[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(ema_34_rising_aligned[i]) or \
           np.isnan(ema_34_falling_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 12h volume > 1.5x 20-period average
        vol_filter = False
        if not np.isnan(vol_ma_20_aligned[i]):
            # Find current 12h bar's volume
            idx_12h = 0
            while idx_12h < len(df_12h) and df_12h.iloc[idx_12h]['open_time'] <= prices.iloc[i]['open_time']:
                idx_12h += 1
            idx_12h -= 1  # last completed 12h bar
            
            if idx_12h >= 0:
                vol_12h_current = df_12h.iloc[idx_12h]['volume']
                vol_filter = vol_12h_current > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for entry: Price above/below lips + trend + volume
            # Long when price above lips (green), price above EMA34, with volume spike
            long_condition = (close[i] > lips_aligned[i]) and \
                             ema_34_rising_aligned[i] and (close[i] > ema_34_aligned[i]) and vol_filter
            # Short when price below lips, price below EMA34, with volume spike
            short_condition = (close[i] < lips_aligned[i]) and \
                              ema_34_falling_aligned[i] and (close[i] < ema_34_aligned[i]) and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses teeth (red) or trend fails
            if (close[i] < teeth_aligned[i]) or (not ema_34_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses teeth or trend fails
            if (close[i] > teeth_aligned[i]) or (not ema_34_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals