# Strategy: 4h_ATRBreakout_12hTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.095 | +24.4% | -12.1% | 81 | PASS |
| ETHUSDT | 0.482 | +51.7% | -11.6% | 78 | PASS |
| SOLUSDT | 0.677 | +89.5% | -26.4% | 62 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.476 | -7.2% | -9.2% | 33 | FAIL |
| ETHUSDT | 0.795 | +18.6% | -6.0% | 28 | PASS |
| SOLUSDT | 0.346 | +11.0% | -9.5% | 24 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h ATR-based volatility breakout with 12h trend filter and volume confirmation.
# Long when price breaks above 12h high + ATR multiplier in uptrend with volume surge.
# Short when price breaks below 12h low - ATR multiplier in downtrend with volume surge.
# Uses 12h EMA(34) for trend direction and 12h ATR(14) for dynamic breakout levels.
# Designed for low trade frequency (15-25/year) to minimize fee drag and capture sustained moves.

name = "4h_ATRBreakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend and volatility
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h EMA(34) for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = ema_34_12h[1:] > ema_34_12h[:-1]  # Rising EMA = uptrend
    trend_up = np.concatenate([[False], trend_up])  # Align with 12h index
    
    # 12h ATR(14) for volatility
    high_low = high_12h - low_12h
    high_close = np.abs(high_12h - np.roll(close_12h, 1))
    low_close = np.abs(low_12h - np.roll(close_12h, 1))
    high_close[0] = high_12h[0] - close_12h[0]  # First value
    low_close[0] = low_12h[0] - close_12h[0]    # First value
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Dynamic breakout levels: 12h high/low ± ATR multiplier
    upper_break = high_12h + (atr_14 * 0.5)
    lower_break = low_12h - (atr_14 * 0.5)
    
    # Align 12h indicators to 4h timeframe
    trend_up_aligned = align_htf_to_ltf(prices, df_12h, trend_up.astype(float))
    upper_break_aligned = align_htf_to_ltf(prices, df_12h, upper_break)
    lower_break_aligned = align_htf_to_ltf(prices, df_12h, lower_break)
    
    # Volume confirmation: 4h volume > 2.0x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for EMA(34)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_up_aligned[i]) or np.isnan(upper_break_aligned[i]) or
            np.isnan(lower_break_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: break above 12h high + ATR in uptrend with volume
            if (trend_up_aligned[i] > 0.5 and  # 12h uptrend
                close[i] > upper_break_aligned[i] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: break below 12h low - ATR in downtrend with volume
            elif (trend_up_aligned[i] <= 0.5 and  # 12h downtrend
                  close[i] < lower_break_aligned[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below 12h low or trend turns down
            if close[i] < lower_break_aligned[i] or trend_up_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above 12h high or trend turns up
            if close[i] > upper_break_aligned[i] or trend_up_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-08 17:47
