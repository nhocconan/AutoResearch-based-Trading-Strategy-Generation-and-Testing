#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot touch + 1d EMA34 trend filter + volume confirmation.
# Long when price touches Camarilla L3 or L4 level in uptrend (EMA34 rising) with volume spike.
# Short when price touches Camarilla H3 or H4 level in downtrend (EMA34 falling) with volume spike.
# Exit when price crosses opposite Camarilla level or EMA34 direction changes.
# Uses Camarilla levels from daily pivot for institutional support/resistance.
# Designed for high-probability reversals in both bull and bear markets with low trade frequency.

name = "4h_Camarilla_Touch_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for price action and volume (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot calculation and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla levels: H4 = close + 1.5*(high-low), H3 = close + 1.125*(high-low), 
    # L3 = close - 1.125*(high-low), L4 = close - 1.5*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ranges and levels
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + 1.5 * range_1d
    camarilla_h3 = close_1d + 1.125 * range_1d
    camarilla_l3 = close_1d - 1.125 * range_1d
    camarilla_l4 = close_1d - 1.5 * range_1d
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_prev = np.roll(ema_34, 1)
    ema_34_prev[0] = ema_34[0]
    ema_34_rising = ema_34 > ema_34_prev
    ema_34_falling = ema_34 < ema_34_prev
    
    # Align all indicators to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_falling)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or \
           np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or \
           np.isnan(ema_34_aligned[i]) or np.isnan(ema_34_rising_aligned[i]) or \
           np.isnan(ema_34_falling_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 4h volume > 1.3x average of last 20 periods
        vol_filter = False
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            if vol_ma_20 > 0:
                vol_filter = volume[i] > 1.3 * vol_ma_20
        
        if position == 0:
            # Look for entry: Camarilla touch + trend + volume
            # Long when price touches L3 or L4 level, EMA34 rising, with volume spike
            long_condition = ((close[i] <= camarilla_l3_aligned[i] * 1.002) and (close[i] >= camarilla_l4_aligned[i] * 0.998)) or \
                            ((close[i] <= camarilla_l4_aligned[i] * 1.002) and (close[i] >= camarilla_l4_aligned[i] * 0.998))
            long_condition = long_condition and ema_34_rising_aligned[i] and vol_filter
            
            # Short when price touches H3 or H4 level, EMA34 falling, with volume spike
            short_condition = ((close[i] >= camarilla_h3_aligned[i] * 0.998) and (close[i] <= camarilla_h4_aligned[i] * 1.002)) or \
                             ((close[i] >= camarilla_h4_aligned[i] * 0.998) and (close[i] <= camarilla_h4_aligned[i] * 1.002))
            short_condition = short_condition and ema_34_falling_aligned[i] and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches H3 level or EMA34 starts falling
            if (close[i] >= camarilla_h3_aligned[i] * 0.998) or (not ema_34_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches L3 level or EMA34 starts rising
            if (close[i] <= camarilla_l3_aligned[i] * 1.002) or (not ema_34_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals