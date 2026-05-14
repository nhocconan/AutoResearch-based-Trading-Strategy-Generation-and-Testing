# Strategy: 6h_Williams_Alligator_1dTrend_VolRegime_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.281 | +34.3% | -11.4% | 157 | PASS |
| ETHUSDT | 0.455 | +50.1% | -19.0% | 158 | PASS |
| SOLUSDT | 1.115 | +192.3% | -18.2% | 155 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.711 | -1.3% | -7.8% | 55 | FAIL |
| ETHUSDT | 0.641 | +16.4% | -8.0% | 44 | PASS |
| SOLUSDT | -0.019 | +4.7% | -11.4% | 50 | FAIL |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator with 1d EMA34 trend filter and ATR-based volatility regime.
# Williams Alligator: Jaw=EMA13(8), Teeth=EMA8(5), Lips=EMA5(3) of median price.
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA34 AND ATR14 > ATR50 (high vol).
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA34 AND ATR14 > ATR50.
# Exit when Alligator alignment reverses OR ATR14 < ATR50 * 0.8 (low vol exit).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~12-37/year) by requiring confluence of Alligator alignment, daily trend, and volatility regime.
# Williams Alligator identifies trending vs ranging markets via convergence/divergence of smoothed SMAs.
# Effective in both bull and bear markets by capturing strong directional moves with trend and volatility filters.

name = "6h_Williams_Alligator_1dTrend_VolRegime_v1"
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Williams Alligator: Median price = (high + low) / 2
    median_price = (high + low) / 2
    
    # Jaw: EMA(13) of median, smoothed 8 periods
    jaw = pd.Series(median_price).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = pd.Series(jaw).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # Teeth: EMA(8) of median, smoothed 5 periods
    teeth = pd.Series(median_price).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = pd.Series(teeth).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Lips: EMA(5) of median, smoothed 3 periods
    lips = pd.Series(median_price).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = pd.Series(lips).ewm(span=3, adjust=False, min_periods=3).mean().values
    
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
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or \
           np.isnan(lips[i]) or np.isnan(atr14[i]) or np.isnan(atr50[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Lips > Teeth > Jaw (bullish alignment), price > 1d EMA34, high volatility regime
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema34_1d_aligned[i] and vol_regime[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: Lips < Teeth < Jaw (bearish alignment), price < 1d EMA34, high volatility regime
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema34_1d_aligned[i] and vol_regime[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator alignment reverses (Lips <= Teeth OR Teeth <= Jaw) OR low volatility regime (ATR14 < ATR50 * 0.8)
            if lips[i] <= teeth[i] or teeth[i] <= jaw[i] or atr14[i] < atr50[i] * 0.8:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: Alligator alignment reverses (Lips >= Teeth OR Teeth >= Jaw) OR low volatility regime (ATR14 < ATR50 * 0.8)
            if lips[i] >= teeth[i] or teeth[i] >= jaw[i] or atr14[i] < atr50[i] * 0.8:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals
```

## Last Updated
2026-05-13 13:24
