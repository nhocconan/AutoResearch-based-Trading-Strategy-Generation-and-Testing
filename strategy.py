#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot-based breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 level, price above 1d EMA34, and volume > 2x 20-period average.
# Short when price breaks below Camarilla S3 level, price below 1d EMA34, and volume > 2x 20-period average.
# Exit when price crosses Camarilla Pivot level or trend fails.
# Uses tight entry conditions to limit trades (target: 20-40/year) and avoid fee drag.
# Camarilla levels provide precise intraday support/resistance; EMA34 filters trend direction.
# Volume spike confirms institutional interest. Designed for 4H timeframe to work in both bull and bear markets.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Volume"
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
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Camarilla levels for each 4h bar (based on previous day's OHLC)
    # Camarilla formulas: 
    # R4 = close + (high - low) * 1.1/2
    # R3 = close + (high - low) * 1.1/4
    # R2 = close + (high - low) * 1.1/6
    # R1 = close + (high - low) * 1.1/12
    # PP = (high + low + close) / 3
    # S1 = close - (high - low) * 1.1/12
    # S2 = close - (high - low) * 1.1/6
    # S3 = close - (high - low) * 1.1/4
    # S4 = close - (high - low) * 1.1/2
    
    # We need previous day's OHLC for current 4h bar's Camarilla levels
    # Since we're using 4h data, we'll use the previous 4h bar's high/low/close
    # This is acceptable as Camarilla is typically calculated from prior period
    
    # Shift to get previous bar's OHLC
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    # Handle first bar
    prev_high_4h[0] = high_4h[0]
    prev_low_4h[0] = low_4h[0]
    prev_close_4h[0] = close_4h[0]
    
    # Calculate Camarilla levels based on previous bar
    rng = prev_high_4h - prev_low_4h
    R3 = prev_close_4h + rng * 1.1 / 4
    S3 = prev_close_4h - rng * 1.1 / 4
    PP = (prev_high_4h + prev_low_4h + prev_close_4h) / 3
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = np.full(len(volume_4h), np.nan)
    for i in range(20, len(volume_4h)):
        vol_ma_20[i] = np.mean(volume_4h[i-20:i])
    
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
    
    # Align all indicators to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_4h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_4h, S3)
    PP_aligned = align_htf_to_ltf(prices, df_4h, PP)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
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
        if np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(PP_aligned[i]) or \
           np.isnan(vol_ma_20_aligned[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(ema_34_rising_aligned[i]) or \
           np.isnan(ema_34_falling_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 4h volume > 2x 20-period average
        vol_filter = False
        if not np.isnan(vol_ma_20_aligned[i]):
            # Find current 4h bar's volume
            idx_4h = 0
            while idx_4h < len(df_4h) and df_4h.iloc[idx_4h]['open_time'] <= prices.iloc[i]['open_time']:
                idx_4h += 1
            idx_4h -= 1  # last completed 4h bar
            
            if idx_4h >= 0:
                vol_4h_current = df_4h.iloc[idx_4h]['volume']
                vol_filter = vol_4h_current > 2.0 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for entry: Price breaks above/below Camarilla S3/R3 + trend + volume spike
            # Long when price breaks above R3, price above EMA34, with volume spike
            long_condition = (close[i] > R3_aligned[i]) and \
                             ema_34_rising_aligned[i] and (close[i] > ema_34_aligned[i]) and vol_filter
            # Short when price breaks below S3, price below EMA34, with volume spike
            short_condition = (close[i] < S3_aligned[i]) and \
                              ema_34_falling_aligned[i] and (close[i] < ema_34_aligned[i]) and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses Camarilla Pivot or trend fails
            if (close[i] < PP_aligned[i]) or (not ema_34_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses Camarilla Pivot or trend fails
            if (close[i] > PP_aligned[i]) or (not ema_34_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals