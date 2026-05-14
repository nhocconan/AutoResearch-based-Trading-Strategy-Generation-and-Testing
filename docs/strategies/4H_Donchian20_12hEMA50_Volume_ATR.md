# Strategy: 4H_Donchian20_12hEMA50_Volume_ATR

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.092 | +24.1% | -16.7% | 109 | PASS |
| ETHUSDT | 0.435 | +50.5% | -14.0% | 104 | PASS |
| SOLUSDT | 1.022 | +185.1% | -30.0% | 96 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.107 | -5.9% | -9.7% | 49 | FAIL |
| ETHUSDT | 0.036 | +5.6% | -9.7% | 39 | PASS |
| SOLUSDT | 0.082 | +6.4% | -15.3% | 34 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above Donchian upper band and close > 12h EMA50 (uptrend) with volume > 2.0x average.
Short when price breaks below Donchian lower band and close < 12h EMA50 (downtrend) with volume > 2.0x average.
Uses 4h timeframe targeting 75-200 total trades over 4 years. EMA50 provides smooth trend filter,
reducing whipsaw in sideways markets. Volume confirmation ensures breakout conviction.
ATR-based stoploss exits when price moves against position by 2.5x ATR(14).
Designed to work in both bull and bear markets by following higher timeframe direction.
"""

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
    
    # Load 4h data for Donchian channels - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channels (20-period) on 4h
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    upper_4h = rolling_max(high_4h, 20)
    lower_4h = rolling_min(low_4h, 20)
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # ATR(14) for stoploss on primary timeframe
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        ema50_val = ema50_12h_aligned[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND price > 12h EMA50 (uptrend) AND volume confirmation
            if (price > upper_val and price > ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower AND price < 12h EMA50 (downtrend) AND volume confirmation
            elif (price < lower_val and price < ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: ATR stoploss OR price breaks below Donchian lower OR trend reversal
                if (price <= entry_price - 2.5 * atr_val or 
                    price < lower_val or 
                    price < ema50_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: ATR stoploss OR price breaks above Donchian upper OR trend reversal
                if (price >= entry_price + 2.5 * atr_val or 
                    price > upper_val or 
                    price > ema50_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_12hEMA50_Volume_ATR"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 02:08
