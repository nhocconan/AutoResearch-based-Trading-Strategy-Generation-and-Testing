#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels for mean reversion and 1w ADX for trend strength filter
# - Uses 1d HTF for Camarilla pivot levels (H3/L3) as key support/resistance
# - Uses 1w HTF for ADX: only trade when ADX > 25 (trending market) to avoid chop
# - In trending markets (ADX > 25): look for mean reversion at extreme Camarilla levels
# - Long when price touches or crosses below L3 and shows reversal signs
# - Short when price touches or crosses above H3 and shows reversal signs
# - Volume confirmation: current 12h volume > 1.5x 20-period average to avoid low-volume false signals
# - Fixed position size 0.25 to control drawdown
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_1w_camarilla_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Typical Price = (high + low + close) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3
    # Range = high - low
    daily_range = high_1d - low_1d
    
    # Camarilla levels
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.0 * (high - low)
    # L3 = close - 1.0 * (high - low)
    # L4 = close - 1.5 * (high - low)
    camarilla_h3 = close_1d + 1.0 * daily_range
    camarilla_l3 = close_1d - 1.0 * daily_range
    
    # Calculate 1w ADX (14 periods)
    # +DM = max(high - previous_high, 0) if high - previous_high > previous_low - low else 0
    # -DM = max(previous_low - low, 0) if previous_low - low > high - previous_high else 0
    # TR = max(high - low, abs(high - previous_close), abs(low - previous_close))
    # +DM14 = smoothed +DM over 14 periods
    # -DM14 = smoothed -DM over 14 periods
    # TR14 = smoothed TR over 14 periods
    # +DI14 = 100 * (+DM14 / TR14)
    # -DI14 = 100 * (-DM14 / TR14)
    # DX = 100 * abs(+DI14 - -DI14) / (+DI14 + -DI14)
    # ADX = smoothed DX over 14 periods
    
    # Calculate components for ADX
    high_diff = np.diff(high_1w, prepend=high_1w[0])
    low_diff = np.diff(low_1w, prepend=low_1w[0])
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    tr3 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3[0] = np.abs(high_1w[0] - close_1w[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values: prev * (period-1)/period + current/period
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    plus_dm_smooth = wilders_smoothing(plus_dm, period)
    minus_dm_smooth = wilders_smoothing(minus_dm, period)
    tr_smooth = wilders_smoothing(tr, period)
    
    # Avoid division by zero
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smoothing(dx, period)
    
    # Align all HTF data to 12h timeframe (wait for completed HTF bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend strength: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Price action: check if price is at or beyond Camarilla extremes
        at_l3 = low[i] <= camarilla_l3_aligned[i]  # Touched or went below L3
        at_h3 = high[i] >= camarilla_h3_aligned[i]  # Touched or went above H3
        
        # Reversal confirmation: price moved back inside the level
        # For long: price closed back above L3 after touching it
        # For short: price closed back below H3 after touching it
        reversed_from_l3 = at_l3 and close[i] > camarilla_l3_aligned[i]
        reversed_from_h3 = at_h3 and close[i] < camarilla_h3_aligned[i]
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions: reversal complete or trend ends
            if reversed_from_l3 or not trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit conditions: reversal complete or trend ends
            if reversed_from_h3 or not trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Entry logic: only in trending markets at extreme levels with reversal signs
            if volume_confirmed and trending:
                if reversed_from_l3:
                    # Price touched L3 and reversed upward: long
                    position = 1
                    signals[i] = position_size
                elif reversed_from_h3:
                    # Price touched H3 and reversed downward: short
                    position = -1
                    signals[i] = -position_size
    
    return signals