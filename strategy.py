#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_12h_camarilla_reversion_v1
# Uses 12h Camarilla levels for mean reversion on 6b timeframe.
# Fades at R3/S3 levels (mean reversion) and breaks out at R4/S4 (trend continuation).
# Uses 6h volume confirmation and ADX filter to distinguish between ranging and trending markets.
# Designed for low trade frequency (target: 12-37 trades/year) to minimize fee drag.
# Works in both bull and bear markets by adapting to regime (reversion in range, breakout in trend).

name = "6h_12h_camarilla_reversion_v1"
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
    
    # Get 12h data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 12h bar
    high_prev = df_12h['high'].shift(1).values
    low_prev = df_12h['low'].shift(1).values
    close_prev = df_12h['close'].shift(1).values
    
    # Camarilla formulas
    range_prev = high_prev - low_prev
    camarilla_r3 = close_prev + range_prev * 1.1 / 2
    camarilla_s3 = close_prev - range_prev * 1.1 / 2
    camarilla_r4 = close_prev + range_prev * 1.1
    camarilla_s4 = close_prev - range_prev * 1.1
    
    # Align to 6h timeframe (12h levels update only after 12h bar closes)
    r3_level = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_level = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    r4_level = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    s4_level = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # ADX trend filter: distinguish between trending and ranging markets
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Plus and Minus Directional Movement
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    
    # Wilder's smoothing function
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smooth(tr, 14)
    plus_dm_smooth = wilders_smooth(plus_dm, 14)
    minus_dm_smooth = wilders_smooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smooth(dx, 14)
    
    # Regime filter: ADX < 25 = ranging (favor mean reversion), ADX > 25 = trending (favor breakout)
    ranging_market = adx < 25
    trending_market = adx >= 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # start after warmup
        # Skip if levels not ready
        if np.isnan(r3_level[i]) or np.isnan(s3_level[i]) or np.isnan(r4_level[i]) or np.isnan(s4_level[i]) or np.isnan(vol_confirm[i]):
            signals[i] = 0.0
            continue
        
        # Mean reversion in ranging markets: fade at R3/S3
        if ranging_market[i]:
            # Long when price touches S3 and shows rejection
            if close[i] <= s3_level[i] and low[i] < s3_level[i] and close[i] > low[i]:
                if position != 1:
                    position = 1
                    signals[i] = 0.25
            # Short when price touches R3 and shows rejection
            elif close[i] >= r3_level[i] and high[i] > r3_level[i] and close[i] < high[i]:
                if position != -1:
                    position = -1
                    signals[i] = -0.25
            # Exit mean reversion position at midpoint
            elif position == 1 and close[i] >= (r3_level[i] + s3_level[i]) / 2:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] <= (r3_level[i] + s3_level[i]) / 2:
                position = 0
                signals[i] = 0.0
            # Hold position
            elif position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        
        # Breakout continuation in trending markets: break at R4/S4
        elif trending_market[i]:
            # Long breakout: price closes above R4
            if close[i] > r4_level[i] and position != 1:
                position = 1
                signals[i] = 0.25
            # Short breakdown: price closes below S4
            elif close[i] < s4_level[i] and position != -1:
                position = -1
                signals[i] = -0.25
            # Exit on opposite breakout
            elif position == 1 and close[i] < s4_level[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] > r4_level[i]:
                position = 0
                signals[i] = 0.0
            # Hold position
            elif position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        
        # Transition between regimes: close positions
        else:
            if position == 1:
                position = 0
                signals[i] = 0.0
            elif position == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals