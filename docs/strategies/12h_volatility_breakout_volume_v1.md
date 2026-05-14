# Strategy: 12h_volatility_breakout_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.337 | +38.0% | -7.9% | 31 | PASS |
| ETHUSDT | -0.210 | +7.1% | -16.2% | 30 | FAIL |
| SOLUSDT | 0.727 | +94.3% | -15.3% | 20 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.397 | +1.6% | -6.0% | 13 | FAIL |
| SOLUSDT | 0.178 | +8.3% | -11.3% | 9 | PASS |

## Code
```python
#!/usr/bin/env python3
# 12h_volatility_breakout_volume_v1
# Hypothesis: 12h strategy using ATR-based volatility breakouts with volume confirmation.
# Breakouts from ATR channels capture strong moves in both bull and bear markets.
# Volume confirmation filters false breakouts. Discrete sizing (0.0, ±0.25) minimizes fee churn.
# Target: 50-150 total trades over 4 years by requiring ATR breakout + volume spike.
# Primary timeframe: 12h, HTF: 1d for trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_volatility_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR for volatility breakout channels (20-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # 20-period ATR multiplier for channel width
    atr_mult = 1.5
    
    # Calculate upper and lower channels
    upper_channel = np.roll(close, 1) + atr_mult * atr
    lower_channel = np.roll(close, 1) - atr_mult * atr
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 1d HTF data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below lower channel OR trend turns bearish
            if close[i] < lower_channel[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above upper channel OR trend turns bullish
            if close[i] > upper_channel[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long breakout: price breaks above upper channel with volume
                if close[i] > upper_channel[i] and close[i] > ema_50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price breaks below lower channel with volume
                elif close[i] < lower_channel[i] and close[i] < ema_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 02:44
