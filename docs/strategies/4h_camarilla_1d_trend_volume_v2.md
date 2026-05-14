# Strategy: 4h_camarilla_1d_trend_volume_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.142 | +8.1% | -15.0% | 258 | FAIL |
| ETHUSDT | 0.624 | +79.6% | -15.5% | 241 | PASS |
| SOLUSDT | 0.698 | +125.2% | -26.7% | 223 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.049 | +5.4% | -17.5% | 85 | PASS |
| SOLUSDT | -0.374 | -4.6% | -23.6% | 77 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 4h_camarilla_1d_trend_volume_v2
# Hypothesis: 4h strategy using 1d Camarilla pivot levels for structure, volume confirmation, and trend filter.
# Uses discrete position sizing (±0.30) to minimize fee churn. Volume > 1.8x 20-period average filters weak moves.
# Long: price above daily pivot + volume spike + close > H3
# Short: price below daily pivot + volume spike + close < L3
# Exits on close < L3 (long) or close > H3 (short). Target: 75-200 total trades over 4 years (19-50/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_1d_trend_volume_v2"
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
    
    # 1d HTF data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First bar: use same day's data (no look-ahead)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla pivot levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    
    # Resistance levels
    H3 = pivot + (range_ * 1.1 / 4)
    H4 = pivot + (range_ * 1.1 / 2)
    # Support levels
    L3 = pivot - (range_ * 1.1 / 4)
    L4 = pivot - (range_ * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(H3_aligned[i]) or np.isnan(H4_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below L3 (trend reversal)
            if close[i] < L3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 (trend reversal)
            if close[i] > H3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.8 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price above pivot + close > H3 (breakout)
                if close[i] > pivot_aligned[i] and close[i] > H3_aligned[i]:
                    position = 1
                    signals[i] = 0.30
                # Short: price below pivot + close < L3 (breakdown)
                elif close[i] < pivot_aligned[i] and close[i] < L3_aligned[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals
```

## Last Updated
2026-04-09 04:58
