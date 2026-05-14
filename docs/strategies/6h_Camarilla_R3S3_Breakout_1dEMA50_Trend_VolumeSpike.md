# Strategy: 6h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.310 | +31.9% | -5.1% | 171 | PASS |
| ETHUSDT | 0.149 | +26.7% | -11.6% | 147 | PASS |
| SOLUSDT | 0.720 | +83.5% | -15.7% | 131 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.379 | +3.1% | -4.0% | 66 | FAIL |
| ETHUSDT | 1.848 | +31.7% | -6.2% | 57 | PASS |
| SOLUSDT | 0.049 | +6.3% | -5.6% | 52 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation
- Long: Close breaks above Camarilla R3 + price > 1d EMA50 (uptrend) + volume > 2.0x 20-period average
- Short: Close breaks below Camarilla S3 + price < 1d EMA50 (downtrend) + volume > 2.0x 20-period average
- Exit: Close retouches Camarilla H3/L3 level OR trend reversal
- Uses discrete position sizing (0.25) to minimize fee churn
- Target: 12-30 trades/year (50-120 over 4 years) to avoid fee drag
- Camarilla levels from 1d provide institutional pivot points; breakouts with volume and trend filter work in both bull and bear markets
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
    
    # Get 1d data for Camarilla pivot and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla levels from 1d OHLC (previous day's close)
    # Camarilla: H5 = C + 1.1*(H-L)/2, H4 = C + 1.1*(H-L)/4, H3 = C + 1.1*(H-L)/6
    # L3 = C - 1.1*(H-L)/6, L4 = C - 1.1*(H-L)/4, L5 = C - 1.1*(H-L)/2
    # Where C = previous day close, H = previous day high, L = previous day low
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 6
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 6
    camarilla_h5 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l5 = prev_close - 1.1 * (prev_high - prev_low) / 2
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low) / 4
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    # Align Camarilla levels to 6h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 needs 50, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or 
            np.isnan(h5_aligned[i]) or 
            np.isnan(l5_aligned[i]) or 
            np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA50
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Camarilla breakout signals with trend filter and volume confirmation
        # Long: Close breaks above H3 + uptrend + volume spike
        # Short: Close breaks below L3 + downtrend + volume spike
        long_signal = (close[i] > h3_aligned[i] and 
                      uptrend and
                      volume[i] > 2.0 * vol_ma[i])
        
        short_signal = (close[i] < l3_aligned[i] and 
                       downtrend and
                       volume[i] > 2.0 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Close retouches H3/L3 level OR trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Close retouches H3 level or trend turns down
                if (close[i] <= h3_aligned[i] or 
                    not uptrend):
                    exit_signal = True
            elif position == -1:
                # Exit short: Close retouches L3 level or trend turns up
                if (close[i] >= l3_aligned[i] or 
                    not downtrend):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-23 18:24
