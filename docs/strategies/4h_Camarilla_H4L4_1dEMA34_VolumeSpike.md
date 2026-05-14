# Strategy: 4h_Camarilla_H4L4_1dEMA34_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.229 | +30.9% | -9.5% | 154 | PASS |
| ETHUSDT | 0.308 | +38.2% | -11.5% | 141 | PASS |
| SOLUSDT | 0.791 | +113.7% | -21.8% | 134 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.000 | -4.0% | -7.8% | 59 | FAIL |
| ETHUSDT | 0.678 | +16.9% | -7.7% | 50 | PASS |
| SOLUSDT | -0.129 | +2.9% | -13.6% | 43 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla H4/L4 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above H4 in 1d uptrend (price > EMA34).
# Short when price breaks below L4 in 1d downtrend (price < EMA34).
# Volume must be > 2.0x 20-period MA to confirm breakout strength.
# Uses discrete sizing 0.25 to minimize fee churn. Target: 75-200 total trades over 4 years.
# This strategy focuses on BTC and ETH as primary targets, using 1d trend filter to avoid counter-trend trades.
# The Camarilla H4/L4 levels provide stronger support/resistance than H3/L3, reducing false breakouts.
# Volume confirmation ensures breakout validity, and the 1d EMA34 ensures we only trade with the higher timeframe trend.

name = "4h_Camarilla_H4L4_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Camarilla levels from 1d OHLC (using previous bar's close)
    close_1d_shifted = np.roll(close_1d, 1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    range_1d = high_1d - low_1d
    h4 = close_1d_shifted + range_1d * 1.1 / 2
    l4 = close_1d_shifted - range_1d * 1.1 / 2
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Volume confirmation: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        trend_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        vol_spike = volume_spike[i]
        
        # Entry logic
        if position == 0:
            # Long: price breaks above H4 AND 1d uptrend AND volume spike
            if close_val > h4_aligned[i] and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L4 AND 1d downtrend AND volume spike
            elif close_val < l4_aligned[i] and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below L4 OR 1d trend turns down
            if close_val < l4_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H4 OR 1d trend turns up
            if close_val > h4_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-03 09:22
