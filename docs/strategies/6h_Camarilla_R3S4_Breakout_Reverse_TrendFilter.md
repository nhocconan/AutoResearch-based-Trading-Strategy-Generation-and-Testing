# Strategy: 6h_Camarilla_R3S4_Breakout_Reverse_TrendFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.087 | +13.9% | -14.6% | 461 | FAIL |
| ETHUSDT | 0.470 | +55.5% | -14.5% | 417 | PASS |
| SOLUSDT | 0.992 | +179.9% | -14.8% | 407 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.293 | +10.6% | -13.2% | 130 | PASS |
| SOLUSDT | 0.575 | +16.5% | -15.7% | 137 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using Camarilla pivot levels from daily timeframe with 6h volume confirmation.
# Camarilla levels (R3/S3 for reversal, R4/S4 for breakout) provide institutional support/resistance.
# Long when price breaks above R4 with volume confirmation, short when breaks below S4.
# Reverse entries at R3/S3 for mean reversion in ranging markets.
# Trend filter using 6h EMA(50) to align with intermediate trend.
# Designed for low trade frequency (15-25/year) to minimize fee drag while capturing both breakout and reversal opportunities.

name = "6h_Camarilla_R3S4_Breakout_Reverse_TrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day's range
    camarilla_r4 = np.zeros_like(close_1d)
    camarilla_r3 = np.zeros_like(close_1d)
    camarilla_s3 = np.zeros_like(close_1d)
    camarilla_s4 = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        # Previous day's high, low, close
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        # Range
        rng = ph - pl
        
        # Camarilla levels
        camarilla_r4[i] = pc + (rng * 1.1 / 2)
        camarilla_r3[i] = pc + (rng * 1.1 / 4)
        camarilla_s3[i] = pc - (rng * 1.1 / 4)
        camarilla_s4[i] = pc - (rng * 1.1 / 2)
    
    # First day has no previous data
    camarilla_r4[0] = camarilla_r3[0] = camarilla_s3[0] = camarilla_s4[0] = np.nan
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 6h EMA(50) for trend filter
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 6h volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA(50)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Breakout entries: price breaks R4 or S4 with volume confirmation
            if close[i] > r4_aligned[i] and vol_confirm[i]:
                # Only take long breakout if above EMA50 (uptrend)
                if close[i] > ema_50[i]:
                    signals[i] = 0.25
                    position = 1
            elif close[i] < s4_aligned[i] and vol_confirm[i]:
                # Only take short breakout if below EMA50 (downtrend)
                if close[i] < ema_50[i]:
                    signals[i] = -0.25
                    position = -1
            # Reverse entries at R3/S3 for mean reversion
            elif close[i] < r3_aligned[i] and close[i] > s3_aligned[i]:
                # In range: sell near R3, buy near S3
                if close[i] > (r3_aligned[i] + s3_aligned[i]) / 2:
                    # Upper half of range - look for rejection at R3
                    if i > 0 and close[i] < close[i-1] and high[i] >= r3_aligned[i]:
                        signals[i] = -0.20
                        position = -1
                else:
                    # Lower half of range - look for bounce at S3
                    if i > 0 and close[i] > close[i-1] and low[i] <= s3_aligned[i]:
                        signals[i] = 0.20
                        position = 1
        elif position == 1:
            # Long exit: break below S3 (mean reversion) or trend turns down
            if close[i] < s3_aligned[i] or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above R3 (mean reversion) or trend turns up
            if close[i] > r3_aligned[i] or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-08 17:27
