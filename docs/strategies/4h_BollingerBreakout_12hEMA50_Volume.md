# Strategy: 4h_BollingerBreakout_12hEMA50_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.308 | +35.6% | -15.8% | 191 | PASS |
| ETHUSDT | 0.199 | +30.8% | -13.6% | 190 | PASS |
| SOLUSDT | 0.772 | +109.1% | -22.5% | 172 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.254 | -6.0% | -9.5% | 66 | FAIL |
| ETHUSDT | 1.089 | +23.8% | -6.9% | 58 | PASS |
| SOLUSDT | -0.320 | -0.4% | -12.1% | 58 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h price closes outside Bollinger Bands (20,2) with volume confirmation (>1.5x 20 EMA volume) and 12h EMA50 trend filter.
# Bollinger breakouts capture volatility expansion; volume confirms institutional interest; 12h EMA50 ensures alignment with higher timeframe trend.
# Works in bull markets (breakouts above upper band) and bear markets (breakdowns below lower band).
# Uses Bollinger Bands for volatility-based breakout detection, which adapts to changing market conditions.
name = "4h_BollingerBreakout_12hEMA50_Volume"
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
    
    # Bollinger Bands (20,2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 trend filter
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: close > upper Bollinger Band + volume confirmation + 12h EMA50 up
            if (price > upper_band[i] and vol_confirm[i] and price > ema_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: close < lower Bollinger Band + volume confirmation + 12h EMA50 down
            elif (price < lower_band[i] and vol_confirm[i] and price < ema_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close crosses below middle Bollinger Band (SMA20)
            if price < sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close crosses above middle Bollinger Band (SMA20)
            if price > sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 01:38
