# Strategy: 4h_12h_1d_pivot_breakout_volume_v3

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.404 | +3.9% | -10.1% | 224 | FAIL |
| ETHUSDT | -0.902 | -19.5% | -22.4% | 209 | FAIL |
| SOLUSDT | 0.375 | +50.0% | -22.9% | 209 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.529 | +14.4% | -9.6% | 62 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_12h_1d_pivot_breakout_volume_v3
Hypothesis: Use 4h price action with 12h pivot levels and 1d trend bias.
Long when 4h price breaks above 12h R1 with 1d bullish trend and volume confirmation.
Short when 4h price breaks below 12h S1 with 1d bearish trend and volume confirmation.
Uses wider volatility-adjusted breakout thresholds to reduce false breakouts and trade frequency.
Target: 15-40 trades/year per symbol (60-160 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_1d_pivot_breakout_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Get 1d data for trend bias
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 12h pivot points (standard floor trader pivots)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h pivot point
    pivot_point = (high_12h + low_12h + close_12h) / 3.0
    # 12h resistance and support levels
    r1 = 2 * pivot_point - low_12h
    s1 = 2 * pivot_point - high_12h
    r2 = pivot_point + (high_12h - low_12h)
    s2 = pivot_point - (high_12h - low_12h)
    
    # Align 12h pivot levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot_point)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2)
    
    # 1d trend bias: close > EMA(50) for bullish, close < EMA(50) for bearish
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish = close_1d > ema_50
    trend_bearish = close_1d < ema_50
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish.astype(float))
    
    # Volume confirmation: volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    # Volatility filter: only trade when ATR(14) > 0.5% of price to avoid choppy markets
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    vol_filter = atr > (close * 0.005)  # ATR > 0.5% of price
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(trend_bullish_aligned[i]) or
            np.isnan(trend_bearish_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below 12h S1 or 1d trend turns bearish
            if close[i] < s1_aligned[i] or trend_bearish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price breaks above 12h R1 or 1d trend turns bullish
            if close[i] > r1_aligned[i] or trend_bullish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above 12h R1 with 1d bullish trend and volume
            if (close[i] > r1_aligned[i] and trend_bullish_aligned[i] > 0.5 and 
                vol_confirm[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 12h S1 with 1d bearish trend and volume
            elif (close[i] < s1_aligned[i] and trend_bearish_aligned[i] > 0.5 and 
                  vol_confirm[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 10:29
