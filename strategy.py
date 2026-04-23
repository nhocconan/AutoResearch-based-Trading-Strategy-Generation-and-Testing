#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation.
Long when Bull Power > 0, Bear Power < 0, ADX > 25 (trending), and volume > 2.0x average.
Short when Bear Power < 0, Bull Power > 0, ADX > 25 (trending), and volume > 2.0x average.
Exit when ADX < 20 (range) or power signals reverse.
Uses 6h timeframe to target ~12-37 trades/year, avoiding fee drag while capturing strong trends.
Elder Ray measures bull/bear strength relative to EMA13. ADX filter ensures we only trade strong trends.
Volume confirmation ensures only high-conviction moves trigger entries.
Works in both bull and bear markets by adapting to trend direction via ADX regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA13 and ADX - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA13 for 1d (Elder Ray base)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Calculate ADX for 1d trend strength
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(values[:period])
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1]/period) + values[i]
        return result
    
    atr_period = 14
    tr_smooth = wilders_smoothing(tr, atr_period)
    dm_plus_smooth = wilders_smoothing(dm_plus, atr_period)
    dm_minus_smooth = wilders_smoothing(dm_minus, atr_period)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    # Handle division by zero
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = wilders_smoothing(dx, atr_period)
    
    # Align Elder Ray and ADX to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, ADX > 25 (trending), volume spike
            if (bull_val > 0 and bear_val < 0 and adx_val > 25 and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, Bull Power > 0, ADX > 25 (trending), volume spike
            elif (bull_val > 0 and bear_val < 0 and adx_val > 25 and vol_current > 2.0 * vol_ma_val):
                # This condition is impossible as written - fixing logic
                # Short when Bear Power < 0 AND Bull Power < 0 (both negative but bear stronger)
                # Actually: Short when Bear Power < 0 AND Bull Power < 0? No.
                # Correct: Short when Bear Power < 0 AND Bull Power > 0 is wrong.
                # Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA
                # For short: we want Bear Power < 0 (low below EMA) AND Bull Power could be anything
                # But we want bearish momentum: Bear Power negative and decreasing?
                # Let's use: Short when Bear Power < 0 AND Bull Power < 0? No.
                # Standard Elder Ray interpretation:
                # - Bull Power > 0 indicates bulls in control
                # - Bear Power < 0 indicates bears in control
                # So: Long when Bull Power > 0 AND Bear Power < 0? Actually both can be true.
                # Better: Trend strength via ADX, direction via price vs EMA.
                # Let's simplify: use price vs EMA13 for direction, Elder Ray for strength confirmation.
                pass  # Will fix below
        
        # Actually, let's use simpler and more effective logic:
        # Long when: close > EMA13 (uptrend), Bull Power increasing, ADX > 25, volume spike
        # Short when: close < EMA13 (downtrend), Bear Power decreasing, ADX > 25, volume spike
        
        # Need EMA13 aligned
        ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
        if np.isnan(ema13_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        ema13_val = ema13_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: price > EMA13 (uptrend), Bull Power > 0 (bulls strong), ADX > 25, volume spike
            if (price > ema13_val and bull_val > 0 and adx_val > 25 and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price < EMA13 (downtrend), Bear Power < 0 (bears strong), ADX > 25, volume spike
            elif (price < ema13_val and bear_val < 0 and adx_val > 25 and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price < EMA13 (trend break) OR ADX < 20 (losing trend) OR power reversal
                if (price < ema13_val or adx_val < 20 or bull_val <= 0):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price > EMA13 (trend break) OR ADX < 20 (losing trend) OR power reversal
                if (price > ema13_val or adx_val < 20 or bear_val >= 0):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_1dADX_Volume"
timeframe = "6h"
leverage = 1.0