# Strategy: 4h_williams_alligator_1d_volume_trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.031 | +19.9% | -15.3% | 148 | PASS |
| ETHUSDT | 0.079 | +21.8% | -18.5% | 142 | PASS |
| SOLUSDT | 1.179 | +283.3% | -26.3% | 112 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.960 | -5.6% | -9.8% | 59 | FAIL |
| ETHUSDT | 0.750 | +21.0% | -8.9% | 51 | PASS |
| SOLUSDT | 0.257 | +10.1% | -15.2% | 38 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Williams Alligator + Daily Volume + Trend Filter
# Hypothesis: Williams Alligator (smoothed MAs) identifies trend absence/presence.
# Jaw (13-period smoothed MA), Teeth (8-period), Lips (5-period). 
# When lines are intertwined (alligator sleeping) = ranging market (avoid).
# When lines diverge with Lips > Teeth > Jaw = uptrend (long).
# When Lips < Teeth < Jaw = downtrend (short).
# Daily trend filter ensures alignment with higher-timeframe momentum.
# Volume spike confirms institutional participation.
# Designed for 4h timeframe with low trade frequency (<50/year).
# Works in bull via Alligator uptrend + volume, in bear via downtrend + volume.
# Target: 40-120 total trades over 4 years (10-30/year).

name = "4h_williams_alligator_1d_volume_trend_v1"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Williams Alligator on 4h close: smoothed moving averages
    # Jaw: 13-period SMMA (smoothed MA) of close
    # Teeth: 8-period SMMA of close
    # Lips: 5-period SMMA of close
    # SMMA formula: today = (yesterday * (period-1) + today's close) / period
    def smoothed_mma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (prev * (period-1) + current) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smoothed_mma(close, 13)
    teeth = smoothed_mma(close, 8)
    lips = smoothed_mma(close, 5)
    
    # Daily trend filter: EMA(20) of daily close
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume confirmation: volume > 2.0x 30-period average (high threshold for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=15).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: Alligator starts sleeping (lines intertwine) or trend turns bearish
            # Sleeping condition: max difference between any two lines < 0.1% of price
            max_diff = max(abs(lips[i] - teeth[i]), abs(teeth[i] - jaw[i]), abs(lips[i] - jaw[i]))
            if max_diff < (0.001 * close[i]) or close[i] < ema_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: Alligator starts sleeping or trend turns bullish
            max_diff = max(abs(lips[i] - teeth[i]), abs(teeth[i] - jaw[i]), abs(lips[i] - jaw[i]))
            if max_diff < (0.001 * close[i]) or close[i] > ema_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Uptrend: Lips > Teeth > Jaw (Alligator awake, mouth open up)
                if lips[i] > teeth[i] > jaw[i] and close[i] > ema_20_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Downtrend: Lips < Teeth < Jaw (Alligator awake, mouth open down)
                elif lips[i] < teeth[i] < jaw[i] and close[i] < ema_20_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 13:39
