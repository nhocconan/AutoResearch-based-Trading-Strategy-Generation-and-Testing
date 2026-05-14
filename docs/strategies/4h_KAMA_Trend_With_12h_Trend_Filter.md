# Strategy: 4h_KAMA_Trend_With_12h_Trend_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.039 | +21.9% | -11.6% | 593 | PASS |
| ETHUSDT | 0.119 | +25.6% | -8.5% | 571 | PASS |
| SOLUSDT | 0.222 | +33.8% | -21.0% | 491 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.747 | -8.9% | -12.0% | 207 | FAIL |
| ETHUSDT | 0.528 | +13.4% | -6.3% | 199 | PASS |
| SOLUSDT | 1.445 | +29.6% | -8.2% | 171 | PASS |

## Code
```python
# 4h_KAMA_Trend_With_12h_Trend_Filter
# Hypothesis: KAMA adapts to market efficiency, reducing whipsaw in chop and catching trends.
# Long when KAMA slope turns up + price > KAMA + volume > 1.5x average + 12h close > 12h EMA34.
# Short when KAMA slope turns down + price < KAMA + volume > 1.5x average + 12h close < 12h EMA34.
# Exit on opposite signal. Position size: ±0.25. Uses 4h primary with 12h trend filter.
# Designed to work in both bull (trend capture) and bear (avoids false signals via 12h filter).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def kama(close, er_length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close))
        er = np.zeros_like(close)
        for i in range(1, len(close)):
            if volatility[i-er_length+1:i+1].sum() > 0:
                er[i] = change[i-er_length+1:i+1].sum() / volatility[i-er_length+1:i+1].sum()
            else:
                er[i] = 0
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_vals = kama(close, 10, 2, 30)
    
    # Calculate KAMA slope (1-period change)
    kama_slope = np.diff(kama_vals, prepend=0)
    
    # Volume confirmation (10-period MA on 4h)
    volume_ma10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 for trend filter
    close_series_12h = pd.Series(close_12h)
    ema34_12h = close_series_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA to 4h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(10, 10, 34)  # KAMA, volume MA10, EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_vals[i]) or 
            np.isnan(kama_slope[i]) or 
            np.isnan(volume_ma10[i]) or 
            np.isnan(ema34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 10-period average
        volume_filter = volume[i] > (1.5 * volume_ma10[i])
        
        # KAMA-based signals
        kama_bullish = kama_slope[i] > 0 and close[i] > kama_vals[i]
        kama_bearish = kama_slope[i] < 0 and close[i] < kama_vals[i]
        
        if position == 0:
            # Long: KAMA bullish + volume filter + 12h uptrend
            if kama_bullish and volume_filter and close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA bearish + volume filter + 12h downtrend
            elif kama_bearish and volume_filter and close[i] < ema34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA turns bearish
            if kama_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA turns bullish
            if kama_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_With_12h_Trend_Filter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-17 08:19
