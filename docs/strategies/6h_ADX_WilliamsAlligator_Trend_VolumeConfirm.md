# Strategy: 6h_ADX_WilliamsAlligator_Trend_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.357 | +1.3% | -13.5% | 39 | FAIL |
| ETHUSDT | 0.127 | +26.1% | -16.9% | 36 | PASS |
| SOLUSDT | 1.133 | +202.5% | -19.0% | 27 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.637 | +17.3% | -10.2% | 13 | PASS |
| SOLUSDT | -0.161 | +1.3% | -12.3% | 13 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h ADX + Williams Alligator Trend + Volume Confirmation
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and alignment,
while ADX > 25 filters for strong trends. Volume confirmation ensures breakout validity.
Works in bull/bear via trend alignment filter. Target: 12-30 trades/year on 6h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Williams Alligator on 1d: SMAs of median price
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    median_price_1d = (df_1d['high'] + df_1d['low']) / 2
    close_1d = df_1d['close'].values
    
    def smma(arr, period):
        """Smoothed Moving Average (similar to Wilder's smoothing)"""
        if len(arr) < period:
            return np.full(len(arr), np.nan)
        result = np.full(len(arr), np.nan)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_1d = smma(median_price_1d.values, 13)
    teeth_1d = smma(median_price_1d.values, 8)
    lips_1d = smma(median_price_1d.values, 5)
    
    # Shift the lines (Jaw: 8, Teeth: 5, Lips: 3)
    jaw_1d = np.roll(jaw_1d, 8)
    teeth_1d = np.roll(teeth_1d, 5)
    lips_1d = np.roll(lips_1d, 3)
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # ADX calculation on 1d (trend strength)
    def calculate_adx(high_arr, low_arr, close_arr, period=14):
        if len(high_arr) < period + 1:
            return np.full(len(high_arr), np.nan)
        
        # True Range
        tr1 = high_arr[1:] - low_arr[1:]
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        dm_plus = np.where((high_arr[1:] - high_arr[:-1]) > (low_arr[:-1] - low_arr[1:]), 
                          np.maximum(high_arr[1:] - high_arr[:-1], 0), 0)
        dm_minus = np.where((low_arr[:-1] - low_arr[1:]) > (high_arr[1:] - high_arr[:-1]), 
                           np.maximum(low_arr[:-1] - low_arr[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed TR, DM+ , DM- (using Wilder's smoothing = EMA with alpha=1/period)
        atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        
        return adx
    
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator and ADX
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # Trend strength filter
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Look for entry signals - require: Alligator alignment + strong trend + volume
            # Long: bullish alignment AND strong trend AND volume spike
            long_entry = bullish_alignment and strong_trend and vol_spike
            # Short: bearish alignment AND strong trend AND volume spike
            short_entry = bearish_alignment and strong_trend and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Alligator loses bullish alignment OR ADX weakens
            if not bullish_alignment or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Alligator loses bearish alignment OR ADX weakens
            if not bearish_alignment or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_WilliamsAlligator_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 07:59
