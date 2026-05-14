# Strategy: 4h_12h_donchian_ema10_breakout_vol_filter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.070 | +23.2% | -9.0% | 128 | PASS |
| ETHUSDT | 0.342 | +39.5% | -10.4% | 126 | PASS |
| SOLUSDT | 0.797 | +113.3% | -21.3% | 122 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.754 | -8.2% | -11.1% | 45 | FAIL |
| ETHUSDT | 0.298 | +9.8% | -8.2% | 40 | PASS |
| SOLUSDT | -0.647 | -4.3% | -16.2% | 43 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter and Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h 10-period EMA (trend filter)
    close_12h = df_12h['close'].values
    ema10_12h = pd.Series(close_12h).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_12h_aligned = align_htf_to_ltf(prices, df_12h, ema10_12h)
    
    # Calculate 12h 20-period high and low for Donchian channels
    high_20 = pd.Series(df_12h['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_12h['low'].values).rolling(window=20, min_periods=20).min().values
    high_20_aligned = align_htf_to_ltf(prices, df_12h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_12h, low_20)
    
    # Calculate 12-period ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr12 = np.full(n, np.nan)
    for i in range(11, n):
        atr12[i] = np.nanmean(tr[i-11:i+1])
    
    # Calculate 20-period ATR EMA for volatility regime
    atr_ema20 = np.full(n, np.nan)
    atr_series = pd.Series(atr12)
    atr_ema20_values = atr_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_ema20[:] = atr_ema20_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema10_12h_aligned[i]) or np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or np.isnan(atr12[i]) or 
            np.isnan(atr_ema20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR12 > 1.0x 20-period ATR EMA (elevated volatility)
        vol_filter = atr12[i] > atr_ema20[i] * 1.0
        
        # Trend filter: price above/below 12h 10 EMA
        price_above_ema10 = close[i] > ema10_12h_aligned[i]
        price_below_ema10 = close[i] < ema10_12h_aligned[i]
        
        # Entry conditions: Donchian breakout in direction of trend with volatility expansion
        long_breakout = close[i] > high_20_aligned[i]  # break above 12h 20-period high
        short_breakout = close[i] < low_20_aligned[i]  # break below 12h 20-period low
        
        long_entry = long_breakout and price_above_ema10 and vol_filter
        short_entry = short_breakout and price_below_ema10 and vol_filter
        
        # Exit conditions: reversal signal or volatility contraction
        long_exit = (close[i] < ema10_12h_aligned[i]) or (atr12[i] < atr_ema20[i] * 0.8)
        short_exit = (close[i] > ema10_12h_aligned[i]) or (atr12[i] < atr_ema20[i] * 0.8)
        
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

name = "4h_12h_donchian_ema10_breakout_vol_filter_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-12 18:37
