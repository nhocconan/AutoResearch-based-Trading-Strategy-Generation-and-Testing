# Strategy: 4h_Camarilla_R1S1_1dEMA34_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.193 | +27.2% | -11.7% | 490 | PASS |
| ETHUSDT | 0.400 | +36.6% | -6.2% | 434 | PASS |
| SOLUSDT | 0.079 | +22.9% | -20.8% | 381 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.926 | -0.4% | -5.6% | 188 | FAIL |
| ETHUSDT | 0.785 | +15.9% | -6.0% | 173 | PASS |
| SOLUSDT | 0.798 | +15.0% | -4.8% | 137 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Long when price breaks above Camarilla R1 AND close > 1d EMA34 (bullish trend)
- Short when price breaks below Camarilla S1 AND close < 1d EMA34 (bearish trend)
- Volume must be > 2.0 * median volume of last 20 bars (volume spike filter to avoid fakeouts)
- Exit on opposite Camarilla breakout or trend reversal (close crosses 1d EMA34)
- Uses 4h primary timeframe with 1d HTF to target 75-200 total trades over 4 years (19-50/year)
- Camarilla levels provide precise intraday support/resistance that work in ranging markets
- 1d EMA34 ensures alignment with higher timeframe trend to avoid whipsaws
- Volume spike filter adapts to changing market conditions, reducing false breakouts
- Designed for BTC/ETH with edge in ranging markets (mean reversion at extremes) and trending markets (breakout continuation)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels (based on previous bar's range)
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    camarilla_s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter: volume > 2.0 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (2.0 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1, trend up (close > EMA34), volume spike
            if close[i] > camarilla_r1[i] and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1, trend down (close < EMA34), volume spike
            elif close[i] < camarilla_s1[i] and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Camarilla S1 OR trend reversal (close < EMA34)
            if close[i] < camarilla_s1[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Camarilla R1 OR trend reversal (close > EMA34)
            if close[i] > camarilla_r1[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-24 03:31
