# Strategy: 6h_camarilla_pivot_volume_trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.166 | +27.9% | -12.2% | 139 | PASS |
| ETHUSDT | 0.217 | +31.8% | -16.0% | 133 | PASS |
| SOLUSDT | 0.758 | +111.5% | -32.8% | 121 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.915 | -3.8% | -7.7% | 49 | FAIL |
| ETHUSDT | -0.736 | -5.4% | -12.4% | 44 | FAIL |
| SOLUSDT | 0.311 | +10.7% | -7.8% | 41 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot + Volume + Trend Filter
# Hypothesis: Camarilla pivot levels from 1d provide institutional support/resistance.
# Price rejecting at R3/S3 with volume confirmation indicates reversal.
# Breakout through R4/S4 with volume indicates continuation.
# Trend filter (1d EMA50) ensures alignment with higher timeframe direction.
# Works in bull/bear as pivot levels adapt to volatility. Targets 20-30 trades/year.

name = "6h_camarilla_pivot_volume_trend_v1"
timeframe = "6h"
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
    
    # 1d Camarilla pivot levels (calculated from previous 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for pivot calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla calculations
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    r3 = pivot + (range_hl * 1.1 / 2)
    s3 = pivot - (range_hl * 1.1 / 2)
    r4 = pivot + (range_hl * 1.1)
    s4 = pivot - (range_hl * 1.1)
    
    # Align to 6s timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(ema50_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below S3 OR trend turns bearish
            if close[i] < s3_6h[i] or close[i] < ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above R3 OR trend turns bullish
            if close[i] > r3_6h[i] or close[i] > ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Fade at R3/S3 with volume spike
            if vol_spike[i]:
                # Sell at R3 rejection
                if close[i] < r3_6h[i] and (i == 50 or close[i-1] >= r3_6h[i-1]):
                    position = -1
                    signals[i] = -0.25
                # Buy at S3 rejection
                elif close[i] > s3_6h[i] and (i == 50 or close[i-1] <= s3_6h[i-1]):
                    position = 1
                    signals[i] = 0.25
            # Breakout through R4/S4 with volume and trend alignment
            if vol_spike[i]:
                # Buy breakout above R4 with bullish trend
                if close[i] > r4_6h[i] and (i == 50 or close[i-1] <= r4_6h[i-1]) and close[i] > ema50_6h[i]:
                    position = 1
                    signals[i] = 0.25
                # Sell breakout below S4 with bearish trend
                elif close[i] < s4_6h[i] and (i == 50 or close[i-1] >= s4_6h[i-1]) and close[i] < ema50_6h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 13:56
