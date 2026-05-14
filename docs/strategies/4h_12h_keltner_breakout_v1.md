# Strategy: 4h_12h_keltner_breakout_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.032 | +18.3% | -10.4% | 125 | FAIL |
| ETHUSDT | 0.141 | +27.0% | -14.6% | 119 | PASS |
| SOLUSDT | 0.764 | +100.1% | -22.1% | 121 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.461 | +12.6% | -6.3% | 42 | PASS |
| SOLUSDT | -0.043 | +4.6% | -15.1% | 41 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 4h_12h_keltner_breakout_v1
# Strategy: 4-hour Keltner breakout with 12-hour trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Price breaks out of Keltner channels during high momentum periods, with 12h trend filter to avoid counter-trend trades.
# Works in bull by capturing breakouts with trend, and in bear by fading false breakouts via trend filter. Volume confirms institutional participation.
# Uses Keltner channels (ATR-based) for dynamic support/resistance, which adapts to volatility regimes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_keltner_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA20 for trend filter
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Calculate ATR for Keltner channels (using 4h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner channels: EMA20 ± 2*ATR
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema_20 + 2.0 * atr
    lower_keltner = ema_20 - 2.0 * atr
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20_12h_aligned[i]) or np.isnan(ema_20[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 12h EMA20
        uptrend_12h = price_close > ema_20_12h_aligned[i]
        downtrend_12h = price_close < ema_20_12h_aligned[i]
        
        # Breakout signals: price breaks Keltner bands with volume
        long_breakout = (price_close > upper_keltner[i]) and vol_spike[i]
        short_breakout = (price_close < lower_keltner[i]) and vol_spike[i]
        
        # Exit when price returns to EMA20 (mean reversion) or opposite breakout
        exit_long = position == 1 and (price_close < ema_20[i])
        exit_short = position == -1 and (price_close > ema_20[i])
        
        # Trading logic: only trade in direction of 12h trend
        if long_breakout and uptrend_12h and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and downtrend_12h and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
```

## Last Updated
2026-04-11 13:38
