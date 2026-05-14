# Strategy: 4h_Camarilla_R1_S1_Breakout_12h_Trend_Volume_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.520 | +49.0% | -11.9% | 191 | PASS |
| ETHUSDT | 0.837 | +82.9% | -12.7% | 174 | PASS |
| SOLUSDT | 0.835 | +126.2% | -20.3% | 174 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.124 | +4.2% | -7.6% | 128 | FAIL |
| ETHUSDT | 0.280 | +9.1% | -7.2% | 46 | PASS |
| SOLUSDT | 0.837 | +17.7% | -7.0% | 51 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12h_Trend_Volume_Filter
Hypothesis: Camarilla R1/S1 levels act as strong support/resistance in mean-reverting markets. 
Breakout above R1 or below S1 with 12h trend confirmation (price above/below 12h EMA34) 
and volume spike (12h volume > 1.5x 20-period average) captures institutional breakouts. 
Designed for 20-50 trades/year to minimize fee drag while working in both bull and bear regimes.
"""

name = "4h_Camarilla_R1_S1_Breakout_12h_Trend_Volume_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 12h data (call once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)

    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values

    # Calculate 12h EMA34 for trend
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)

    # Calculate 12h volume average (20-period)
    vol_avg_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)

    # Calculate daily Camarilla levels (using 1d data for more stable levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla R1 and S1 levels
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe with 1-day delay (need previous day's data)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1, additional_delay_bars=1)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        ema34_val = ema34_12h_aligned[i]
        vol_avg_val = vol_avg_20_12h_aligned[i]
        vol_12h_val = volume_12h[i // 3]  # 3x 4h bars in 12h

        if np.isnan(camarilla_r1_val) or np.isnan(camarilla_s1_val) or np.isnan(ema34_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above R1 + 12h uptrend + volume spike
            if close[i] > camarilla_r1_val and close[i] > ema34_val and vol_12h_val > vol_avg_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 + 12h downtrend + volume spike
            elif close[i] < camarilla_s1_val and close[i] < ema34_val and vol_12h_val > vol_avg_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below S1 or trend reversal
            if close[i] < camarilla_s1_val or close[i] < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above R1 or trend reversal
            if close[i] > camarilla_r1_val or close[i] > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-12 21:33
