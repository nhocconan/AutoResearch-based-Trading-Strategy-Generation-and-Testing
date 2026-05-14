# Strategy: 4h_ema20_rsi14_atr14_filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.008 | +16.2% | -23.4% | 356 | FAIL |
| ETHUSDT | 0.372 | +50.5% | -19.2% | 352 | PASS |
| SOLUSDT | 1.011 | +228.9% | -29.4% | 391 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.122 | +7.1% | -11.8% | 135 | PASS |
| SOLUSDT | 0.459 | +16.1% | -12.1% | 114 | PASS |

## Code
```python
#!/usr/bin/env python3
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
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period EMA on 1d (trend filter)
    close_1d_series = pd.Series(close_1d)
    ema_20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 14-period RSI on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Align indicators to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(atr_14_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or
            np.isnan(rsi_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA20
        above_ema = close[i] > ema_20_aligned[i]
        below_ema = close[i] < ema_20_aligned[i]
        
        # RSI conditions: avoid extreme levels
        rsi_not_overbought = rsi_14_aligned[i] < 75
        rsi_not_oversold = rsi_14_aligned[i] > 25
        
        # Volatility filter: require sufficient ATR
        vol_ok = atr_14_aligned[i] > 0
        
        # Entry conditions
        long_entry = above_ema and rsi_not_overbought and vol_ok
        short_entry = below_ema and rsi_not_oversold and vol_ok
        
        # Exit conditions: opposite signal or RSI extreme
        exit_long = position == 1 and (below_ema or rsi_14_aligned[i] > 80)
        exit_short = position == -1 and (above_ema or rsi_14_aligned[i] < 20)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_ema20_rsi14_atr14_filter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-13 12:03
