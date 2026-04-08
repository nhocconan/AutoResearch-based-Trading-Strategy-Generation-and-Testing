# Strategy: 6h_donchian20_1d_ema_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.007 | +19.2% | -14.0% | 133 | FAIL |
| ETHUSDT | -0.372 | -4.0% | -19.0% | 147 | FAIL |
| SOLUSDT | 0.712 | +105.0% | -22.0% | 134 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.126 | +7.3% | -9.5% | 44 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation
# Uses Donchian channel breakouts for trend following, confirmed by 1d EMA trend direction
# and volume above 20-period average. Includes ATR-based stoploss to limit drawdowns.
# Designed for moderate trade frequency (target: 12-37 trades/year) to balance signal quality
# and fee efficiency. Works in bull markets via breakouts and in bear markets via
# short breakdowns with trend filter alignment.

name = "6h_donchian20_1d_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Breakout conditions
        long_breakout = close[i] > highest_high[i-1]  # Break above previous high
        short_breakout = close[i] < lowest_low[i-1]   # Break below previous low
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Exit conditions: reverse signal or stoploss
        if position == 1:  # Long position
            # Exit on reverse breakout or stoploss (2*ATR below entry)
            if short_breakout or close[i] <= lowest_low[i-1] + 2 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on reverse breakout or stoploss (2*ATR above entry)
            if long_breakout or close[i] >= highest_high[i-1] - 2 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: bullish breakout with uptrend and volume confirmation
            if long_breakout and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: bearish breakout with downtrend and volume confirmation
            elif short_breakout and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
```

## Last Updated
2026-04-07 05:22
