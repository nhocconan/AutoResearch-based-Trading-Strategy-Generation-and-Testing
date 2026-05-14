# Strategy: 4h_1d_donchian_ema10_breakout_vol_filter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.021 | +19.4% | -11.6% | 83 | FAIL |
| ETHUSDT | 0.242 | +32.8% | -13.4% | 73 | PASS |
| SOLUSDT | 1.353 | +239.5% | -17.3% | 72 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.176 | +8.0% | -8.2% | 25 | PASS |
| SOLUSDT | -0.539 | -2.6% | -9.3% | 25 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d 10-period EMA (trend filter)
    close_1d = df_1d['close'].values
    ema10_1d = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_1d_aligned = align_htf_to_ltf(prices, df_1d, ema10_1d)
    
    # Calculate 1d 20-period high and low for Donchian channels
    high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Calculate 14-period ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = np.full(n, np.nan)
    for i in range(13, n):
        atr14[i] = np.nanmean(tr[i-13:i+1])
    
    # Calculate 20-period ATR EMA for volatility regime
    atr_ema20 = np.full(n, np.nan)
    atr_series = pd.Series(atr14)
    atr_ema20_values = atr_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_ema20[:] = atr_ema20_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema10_1d_aligned[i]) or np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or np.isnan(atr14[i]) or 
            np.isnan(atr_ema20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR14 > 1.0x 20-period ATR EMA (elevated volatility)
        vol_filter = atr14[i] > atr_ema20[i] * 1.0
        
        # Trend filter: price above/below 1d 10 EMA
        price_above_ema10 = close[i] > ema10_1d_aligned[i]
        price_below_ema10 = close[i] < ema10_1d_aligned[i]
        
        # Entry conditions: Donchian breakout in direction of trend with volatility expansion
        long_breakout = close[i] > high_20_aligned[i]  # break above 1d 20-period high
        short_breakout = close[i] < low_20_aligned[i]  # break below 1d 20-period low
        
        long_entry = long_breakout and price_above_ema10 and vol_filter
        short_entry = short_breakout and price_below_ema10 and vol_filter
        
        # Exit conditions: reversal signal or volatility contraction
        long_exit = (close[i] < ema10_1d_aligned[i]) or (atr14[i] < atr_ema20[i] * 0.8)
        short_exit = (close[i] > ema10_1d_aligned[i]) or (atr14[i] < atr_ema20[i] * 0.8)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_ema10_breakout_vol_filter_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-12 18:46
