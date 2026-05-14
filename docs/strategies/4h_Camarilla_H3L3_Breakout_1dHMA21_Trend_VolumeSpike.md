# Strategy: 4h_Camarilla_H3L3_Breakout_1dHMA21_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.174 | +28.5% | -13.0% | 239 | PASS |
| ETHUSDT | 0.291 | +37.6% | -13.9% | 232 | PASS |
| SOLUSDT | 0.531 | +72.5% | -34.1% | 198 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.403 | -7.3% | -11.6% | 94 | FAIL |
| ETHUSDT | 0.135 | +7.4% | -12.4% | 80 | PASS |
| SOLUSDT | -0.127 | +3.0% | -14.7% | 65 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h Camarilla H3L3 Breakout with 1d HMA21 Trend and Volume Spike
Hypothesis: Camarilla pivot levels (H3/L3) act as strong intraday support/resistance. Breakouts above H3 or below L3 with 
volume confirmation and aligned 1d HMA21 trend capture institutional moves. The 1d HMA21 reduces lag vs EMA while smoothing 
noise, providing reliable trend direction. Volume spike confirms participation. Designed for low-moderate trade frequency 
(19-50/year) on 4h timeframe to work in both bull and bear markets via trend following.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=np.float64)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = pd.Series(series).ewm(span=half_period, adjust=False).mean()
    # WMA of full period
    wma_full = pd.Series(series).ewm(span=period, adjust=False).mean()
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    # Final HMA: WMA of raw_hma with sqrt_period
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean()
    return hma.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HMA21 trend and Camarilla pivots (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 21-period HMA on 1d close for trend
    hma_21_1d = calculate_hma(df_1d['close'].values, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate Camarilla pivots for each 1d bar: based on previous day's high, low, close
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla pivots for each 1d bar: based on previous day's high, low, close
    # We need to shift to avoid look-ahead: use previous day's data to calculate today's levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla formulas:
    # H3 = close + (high - low) * 1.1/6
    # L3 = close - (high - low) * 1.1/6
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 6
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 6
    
    # Align to LTF (4h) - no extra delay needed as pivots are based on completed 1d bar
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for HMA, volume MA, and to avoid NaN from shift
    start_idx = max(21, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(hma_21_1d_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        hma_trend = hma_21_1d_aligned[i]
        h3_level = camarilla_h3_aligned[i]
        l3_level = camarilla_l3_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 resistance AND volume spike AND price > 1d HMA21 (uptrend)
            long_entry = (curr_close > h3_level) and vol_spike and (curr_close > hma_trend)
            # Short: price breaks below L3 support AND volume spike AND price < 1d HMA21 (downtrend)
            short_entry = (curr_close < l3_level) and vol_spike and (curr_close < hma_trend)
            
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
            # Exit: price crosses below L3 support (broken support) OR price crosses below HMA (trend change)
            if (curr_close < l3_level) or (curr_close < hma_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above H3 resistance (broken resistance) OR price crosses above HMA (trend change)
            if (curr_close > h3_level) or (curr_close > hma_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dHMA21_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 05:29
