# Strategy: 6h_ElderRay_DailyTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.077 | +23.4% | -8.9% | 185 | PASS |
| ETHUSDT | 0.019 | +19.4% | -17.2% | 165 | PASS |
| SOLUSDT | 1.062 | +177.0% | -21.1% | 130 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.693 | -1.7% | -6.5% | 72 | FAIL |
| ETHUSDT | 0.196 | +8.6% | -8.9% | 58 | PASS |
| SOLUSDT | -0.226 | +0.7% | -14.5% | 53 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray (Bull/Bear Power) with 1-day trend filter and volume confirmation
# Long when Bull Power > 0 (close > EMA13) + daily EMA(50) uptrend + volume spike
# Short when Bear Power < 0 (close < EMA13) + daily EMA(50) downtrend + volume spike
# Elder Ray measures bull/bear strength relative to EMA, effective in trending and ranging markets
# Daily trend filter ensures alignment with higher timeframe momentum
# Volume spike confirms institutional participation
# Targets 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "6h_ElderRay_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    daily_close = df_1d['close'].values
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate EMA(13) for Elder Ray (6x timeframe: 6h * 13 = 78h ≈ 3.25 days)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: high - EMA13
    bear_power = low - ema13   # Bear Power: low - EMA13
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1d_val = ema50_1d_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Bull Power > 0 + daily uptrend + volume spike
            if bull > 0 and close[i] > ema50_1d_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0 + daily downtrend + volume spike
            elif bear < 0 and close[i] < ema50_1d_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR daily trend turns down
            if bull <= 0 or close[i] < ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 OR daily trend turns up
            if bear >= 0 or close[i] > ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-08 13:02
