# Strategy: 6h_1dKeltner_EMA50_Volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.024 | +0.8% | -10.2% | 143 | FAIL |
| ETHUSDT | 0.075 | +23.4% | -4.1% | 140 | PASS |
| SOLUSDT | 0.354 | +38.6% | -10.1% | 157 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.411 | +9.3% | -2.4% | 57 | PASS |
| SOLUSDT | -0.673 | +1.6% | -4.8% | 46 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day Keltner Channel with trend filter and volume confirmation
# Long when price breaks above upper Keltner channel with price > 50-period EMA and volume > 1.5x average
# Short when price breaks below lower Keltner channel with price < 50-period EMA and volume > 1.5x average
# Uses daily ATR-based channels for dynamic support/resistance, EMA50 for trend filter, volume for confirmation
# Designed to work in bull markets via breakouts above resistance and in bear markets via breakdowns below support
# Target: 20-40 trades per year (80-160 over 4 years) with 0.25 position sizing

name = "6h_1dKeltner_EMA50_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA50 on 6h close (needs 50 bars)
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate 1-day Keltner Channel (20-period EMA, 2x ATR)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # EMA20 of close
    ema20_1d = pd.Series(df_1d['close']).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # ATR calculation (14-period)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Keltner Bands
    upper_keltner = ema20_1d + (2 * atr14)
    lower_keltner = ema20_1d - (2 * atr14)
    
    # Align Keltner levels to 6h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    
    # Volume confirmation: >1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.5 * vol_ma_50)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema50[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above upper Keltner with uptrend and volume confirmation
            if close[i] > upper_aligned[i] and close[i] > ema50[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below lower Keltner with downtrend and volume confirmation
            elif close[i] < lower_aligned[i] and close[i] < ema50[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Keltner (support break)
            if close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Keltner (resistance break)
            if close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-06 22:21
