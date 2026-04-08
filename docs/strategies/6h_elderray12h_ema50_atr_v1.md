# Strategy: 6h_elderray12h_ema50_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.666 | -15.2% | -25.1% | 1671 | DISCARD |
| ETHUSDT | -0.022 | +14.2% | -28.3% | 1647 | DISCARD |
| SOLUSDT | 0.644 | +100.5% | -29.4% | 1675 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.482 | +15.6% | -8.8% | 492 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA(50) trend filter and ATR volatility filter.
# Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Enter long when Bull Power > 0 and EMA50 rising, short when Bear Power < 0 and EMA50 falling.
# Use ATR(10) to filter low volatility periods (ATR < 0.5 * ATRMA50).
# Exit on opposite Elder Ray signal or when price crosses 12h EMA(50).
# Designed to work in both bull (capture trends) and bear (fade bounces) markets.
# Target: 75-200 total trades over 4 years (19-50/year) with controlled risk.

name = "6h_elderray12h_ema50_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Elder Ray components (EMA13 for power calculation)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # ATR(10) for volatility filter
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]  # first period
    atr = pd.Series(tr).ewm(span=10, adjust=False).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(atr[i]) or np.isnan(atr_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR > 0.5 * ATR_MA (avoid choppy low-vol periods)
        vol_filter = atr[i] > 0.5 * atr_ma[i]
        
        if position == 1:  # long position
            # Exit: Bear Power > 0 (market turning bearish) OR price crosses below EMA50
            if bear_power[i] > 0 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bull Power < 0 (market turning bullish) OR price crosses above EMA50
            if bull_power[i] < 0 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Elder Ray signal + EMA50 trend + volatility filter
            if vol_filter:
                if bull_power[i] > 0 and close[i] > ema_50_aligned[i]:
                    # Bullish power with price above EMA50: long
                    signals[i] = 0.25
                    position = 1
                elif bear_power[i] < 0 and close[i] < ema_50_aligned[i]:
                    # Bearish power with price below EMA50: short
                    signals[i] = -0.25
                    position = -1
    
    return signals
```

## Last Updated
2026-04-07 04:13
