# Strategy: 6h_ElderRay_BullBearPower_1dTrend_VolRegime_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.075 | +23.5% | -13.6% | 256 | PASS |
| ETHUSDT | 0.210 | +30.7% | -11.4% | 268 | PASS |
| SOLUSDT | 0.469 | +61.3% | -18.1% | 266 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.769 | -0.7% | -6.2% | 88 | FAIL |
| ETHUSDT | 0.783 | +17.1% | -7.9% | 82 | PASS |
| SOLUSDT | -0.498 | -1.0% | -14.8% | 79 | FAIL |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and ATR-based volatility regime.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low. Enter long when Bull Power > 0 and Bear Power < 0 with price above 1d EMA50 and ATR(14) > ATR(50) (high volatility regime).
# Enter short when Bear Power > 0 and Bull Power < 0 with price below 1d EMA50 and ATR(14) > ATR(50).
# Exit when Elder Ray signals reverse or ATR(14) < ATR(50) * 0.8 (low volatility exit).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~12-37/year) by requiring confluence of Elder Ray alignment, HTF trend, and volatility regime.
# Elder Ray measures bull/bear strength relative to EMA13, effective in both trending and volatile markets.
# Volatility regime filter (ATR14 > ATR50) ensures trading only during sufficient price action, reducing false signals in low-volatility periods.

name = "6h_ElderRay_BullBearPower_1dTrend_VolRegime_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate EMA(13) for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # ATR(14) and ATR(50) for volatility regime
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Volatility regime: ATR14 > ATR50 (high volatility)
    vol_regime = atr14 > atr50
    
    # Track entry price for stoploss (optional, using signal reversal as primary exit)
    entry_price = np.full(n, np.nan)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(ema13[i]) or \
           np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or \
           np.isnan(atr14[i]) or np.isnan(atr50[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0, Bear Power < 0, price above 1d EMA50, high volatility regime
            if bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema50_1d_aligned[i] and vol_regime[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: Bear Power > 0, Bull Power < 0, price below 1d EMA50, high volatility regime
            elif bear_power[i] > 0 and bull_power[i] < 0 and close[i] < ema50_1d_aligned[i] and vol_regime[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Elder Ray reverses (Bear Power >= 0) OR low volatility regime (ATR14 < ATR50 * 0.8)
            if bear_power[i] >= 0 or atr14[i] < atr50[i] * 0.8:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: Elder Ray reverses (Bull Power >= 0) OR low volatility regime (ATR14 < ATR50 * 0.8)
            if bull_power[i] >= 0 or atr14[i] < atr50[i] * 0.8:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals
```

## Last Updated
2026-05-13 13:19
