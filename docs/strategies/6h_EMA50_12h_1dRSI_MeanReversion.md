# Strategy: 6h_EMA50_12h_1dRSI_MeanReversion

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.329 | -1.3% | -17.0% | 245 | FAIL |
| ETHUSDT | 0.307 | +42.1% | -19.9% | 239 | PASS |
| SOLUSDT | 0.657 | +106.3% | -31.8% | 268 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.381 | +12.7% | -11.2% | 74 | PASS |
| SOLUSDT | 0.547 | +18.0% | -11.5% | 71 | PASS |

## Code
```python
#\usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h price crossing 12h EMA50 with 1d RSI(14) mean reversion filter.
# Uses 12h EMA for trend direction, 1d RSI to avoid overbought/oversold extremes,
# and price crossovers for entries. Designed to capture trend continuations
# while avoiding counter-trend trades in ranging markets. Target: 15-30 trades/year.
name = "6h_EMA50_12h_1dRSI_MeanReversion"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 12h close
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period RSI on 1d close
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need 50 periods for EMA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(rsi_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema = ema_50_12h_aligned[i]
        rsi_val = rsi_aligned[i]
        
        if position == 0:
            # Enter long: price crosses above EMA50 AND RSI < 70 (not overbought)
            if price > ema and rsi_val < 70:
                signals[i] = 0.25
                position = 1
            # Enter short: price crosses below EMA50 AND RSI > 30 (not oversold)
            elif price < ema and rsi_val > 30:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA50 OR RSI > 70 (overbought)
            if price < ema or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA50 OR RSI < 30 (oversold)
            if price > ema or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 02:26
