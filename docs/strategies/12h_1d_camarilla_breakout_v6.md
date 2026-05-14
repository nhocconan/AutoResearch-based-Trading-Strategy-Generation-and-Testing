# Strategy: 12h_1d_camarilla_breakout_v6

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.073 | +23.4% | -7.7% | 139 | PASS |
| ETHUSDT | 0.469 | +40.3% | -7.3% | 106 | PASS |
| SOLUSDT | -0.020 | +20.3% | -11.0% | 47 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.730 | -1.2% | -9.7% | 53 | FAIL |
| ETHUSDT | 1.292 | +21.7% | -5.5% | 36 | PASS |

## Code
```python
#!/usr/bin/env python3
# 12h_1d_camarilla_breakout_v6
# Hypothesis: Breakout above/below 1d Camarilla H3/L3 levels on 12h chart with volume confirmation.
# Uses 1d trend filter: only take long trades when price > 1d EMA(50), only short trades when price < 1d EMA(50).
# Exit when price returns to opposite side of pivot point (mean reversion).
# Target: 12-37 trades/year (50-150 total over 4 years) with strict entry conditions.
# Works in both bull and bear markets due to breakout logic + trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_v6"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for volatility filter
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = 0.9 * atr[i-1] + 0.1 * tr[i]  # Wilder's smoothing
    
    # Load 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for previous day (using H3/L3 for tighter entries)
    ph = df_1d['high'].values  # previous day high
    pl = df_1d['low'].values   # previous day low
    pc = df_1d['close'].values # previous day close
    
    range_1d = ph - pl
    # H3 = close + (high - low) * 1.1/4
    # L3 = close - (high - low) * 1.1/4
    h3 = pc + range_1d * 1.1 / 4
    l3 = pc - range_1d * 1.1 / 4
    # Pivot point = (high + low + close) / 3
    pp = (ph + pl + pc) / 3
    
    # Align Camarilla levels to 12h timeframe (wait for previous day's close)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Load 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = np.zeros_like(close_1d, dtype=float)
    ema_1d[0] = close_1d[0]
    alpha = 2.0 / (50 + 1)
    for i in range(1, len(close_1d)):
        ema_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_1d[i-1]
    
    # Align 1d EMA to 12h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation - 20 period average
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(atr[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely high volatility
        vol_filter = atr[i] < 0.05 * close[i]  # ATR less than 5% of price (tighter)
        
        # Volume confirmation: current volume > 1.5x 20-period average (tighter)
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        # Trend filter: price > 1d EMA for longs, price < 1d EMA for shorts
        trend_long = close[i] > ema_1d_aligned[i]
        trend_short = close[i] < ema_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below pivot point (mean reversion)
            if close[i] < pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price closes above pivot point (mean reversion)
            if close[i] > pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Enter long: price closes above H3 with volume confirmation, volatility filter, and trend filter
            if close[i] > h3_aligned[i] and vol_ok and vol_filter and trend_long:
                position = 1
                signals[i] = 0.30
            # Enter short: price closes below L3 with volume confirmation, volatility filter, and trend filter
            elif close[i] < l3_aligned[i] and vol_ok and vol_filter and trend_short:
                position = -1
                signals[i] = -0.30
    
    return signals
```

## Last Updated
2026-04-09 07:24
