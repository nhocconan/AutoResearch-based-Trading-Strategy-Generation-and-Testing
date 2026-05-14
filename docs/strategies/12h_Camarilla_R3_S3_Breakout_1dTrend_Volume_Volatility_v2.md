# Strategy: 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Volatility_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.004 | +20.6% | -8.0% | 129 | PASS |
| ETHUSDT | 0.063 | +22.7% | -11.1% | 105 | PASS |
| SOLUSDT | 0.281 | +39.8% | -26.5% | 107 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.521 | +1.4% | -5.8% | 45 | FAIL |
| ETHUSDT | 0.305 | +9.9% | -5.2% | 39 | PASS |
| SOLUSDT | -0.312 | +1.0% | -11.4% | 35 | FAIL |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter, volume confirmation (1.6x MA20), and ATR volatility filter (ATR14 > 0.35 * ATR50).
# Enters long when price breaks above Camarilla R3 level with 1d bullish trend (close > EMA34), volume > 1.6x MA20, and sufficient volatility.
# Enters short when price breaks below Camarilla S3 level with 1d bearish trend (close < EMA34), volume > 1.6x MA20, and sufficient volatility.
# Exits when price reverts to the Camarilla pivot point or ATR-based stoploss (2.0 * ATR14 from entry).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~12-37/year) by requiring strict confluence: price breakout + HTF trend + volume spike + volatility filter.
# Camarilla R3/S3 levels provide strong intraday support/resistance, while 1d EMA34 filter ensures alignment with higher timeframe momentum.
# Volume threshold (1.6x) and volatility filter (0.35x) reduce false breakouts, improving signal quality in both bull and bear markets.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Volatility_v2"
timeframe = "12h"
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
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate EMA(34) on 1d close
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla levels are calculated using previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values  # previous day high
    prev_low = df_1d['low'].shift(1).values    # previous day low
    prev_close = df_1d['close'].shift(1).values # previous day close
    
    # Calculate Camarilla levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r3 = pivot + (range_hl * 1.1 / 4.0)  # Resistance 3
    s3 = pivot - (range_hl * 1.1 / 4.0)  # Support 3
    r4 = pivot + (range_hl * 1.1 / 2.0)  # Resistance 4
    s4 = pivot - (range_hl * 1.1 / 2.0)  # Support 4
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume filter: current volume > 1.6x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.6)
    
    # ATR(14) and ATR(50) for volatility filter
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr14 > (0.35 * atr50)  # avoid low volatility breakouts
    
    # Track entry price for ATR-based stoploss
    entry_price = np.full(n, np.nan)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pivot_aligned[i]) or \
           np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i]) or \
           np.isnan(atr14[i]) or np.isnan(atr50[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 with 1d bullish trend, volume spike, and sufficient volatility
            if close[i] > r3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: Price breaks below Camarilla S3 with 1d bearish trend, volume spike, and sufficient volatility
            elif close[i] < s3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to Camarilla pivot (mean reversion) OR ATR stoploss hit
            if close[i] < pivot_aligned[i] or close[i] < entry_price[i-1] - 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: Price reverts to Camarilla pivot (mean reversion) OR ATR stoploss hit
            if close[i] > pivot_aligned[i] or close[i] > entry_price[i-1] + 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals
```

## Last Updated
2026-05-13 12:51
