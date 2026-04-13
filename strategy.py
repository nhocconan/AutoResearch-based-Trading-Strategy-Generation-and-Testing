#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + ADX regime filter
    # Long when: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 (strong trend)
    # Short when: Bear Power < 0 AND Bull Power > 0 AND ADX > 25 (strong trend)
    # Exit when: ADX < 20 (trend weakening) OR power signals reverse
    # Uses Elder Ray to measure bull/bear power relative to EMA13, ADX for trend strength.
    # Works in bull (strong uptrend with bull power dominance) and bear (strong downtrend with bear power dominance).
    # Avoids ranging markets (ADX < 20) and reduces whipsaw.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate ADX (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+ , DM- (using Wilder's smoothing = EMA with alpha=1/period)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr = WilderSmooth(tr, 14)
    dm_plus_smooth = WilderSmooth(dm_plus, 14)
    dm_minus_smooth = WilderSmooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = WilderSmooth(dx, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Start after warmup period for all indicators
    start_idx = max(13, 14 + 14 + 14)  # EMA13 + TR smoothing + DM smoothing + DX smoothing
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = (bull_power[i] > 0) and (bear_power[i] < 0) and (adx[i] > 25) and (position != 1)
        short_entry = (bear_power[i] < 0) and (bull_power[i] > 0) and (adx[i] > 25) and (position != -1)
        
        # Exit conditions: trend weakening or power reversal
        exit_long = (position == 1) and ((adx[i] < 20) or (bull_power[i] <= 0) or (bear_power[i] >= 0))
        exit_short = (position == -1) and ((adx[i] < 20) or (bull_power[i] >= 0) or (bear_power[i] <= 0))
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0