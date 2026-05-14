# Strategy: 6H_WILLIAMS_R_MULTIPLIER_BREAKOUT

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.253 | +12.7% | -11.9% | 43 | FAIL |
| ETHUSDT | 0.136 | +26.1% | -9.2% | 36 | PASS |
| SOLUSDT | 0.761 | +93.9% | -21.1% | 40 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.304 | +20.9% | -4.3% | 7 | PASS |
| SOLUSDT | -0.342 | +1.6% | -6.6% | 8 | FAIL |

## Code
```python
# 6H_WILLIAMS_R_MULTIPLIER_BREAKOUT
# Hypothesis: Williams %R overbought/oversold levels (80/20) combined with volatility multiplier (ATR) and 1d trend filter.
# Long when %R crosses above 20 from below with ATR expansion and price above 1d EMA; short when %R crosses below 80 from above with ATR expansion and price below 1d EMA.
# Exit when %R returns to opposite extreme or trend invalidates. Designed to capture momentum reversals in ranging markets while filtering weak signals.
# Targets 20-40 trades/year to minimize fee drain with high-probability setups.

name = "6H_WILLIAMS_R_MULTIPLIER_BREAKOUT"
timeframe = "6h"
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
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    willr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # ATR for volatility filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=10, min_periods=10).mean().values  # 10-period ATR MA for expansion
    
    # 1d EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    pclose = df_1d['close'].values
    ema1d = pd.Series(pclose).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema1d_aligned = align_htf_to_ltf(prices, df_1d, ema1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(willr[i]) or np.isnan(atr[i]) or np.isnan(atr_ma[i]) or np.isnan(ema1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR > 1.2x ATR MA (expansion)
        vol_expansion = atr[i] > atr_ma[i] * 1.2
        
        if position == 0:
            # LONG: %R crosses above 20 from below with vol expansion and uptrend
            if willr[i] > -20 and willr[i-1] <= -20 and vol_expansion and close[i] > ema1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: %R crosses below 80 from above with vol expansion and downtrend
            elif willr[i] < -80 and willr[i-1] >= -80 and vol_expansion and close[i] < ema1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: %R returns to oversold or trend breaks
            if willr[i] < -80 or close[i] < ema1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: %R returns to overbought or trend breaks
            if willr[i] > -20 or close[i] > ema1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 09:27
