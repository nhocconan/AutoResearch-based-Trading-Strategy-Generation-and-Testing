# Strategy: 4H_Camarilla_R4S4_Breakout_1dEMA34_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.508 | +36.7% | -6.9% | 266 | PASS |
| ETHUSDT | 0.078 | +23.6% | -6.5% | 246 | PASS |
| SOLUSDT | 0.290 | +36.2% | -9.1% | 221 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.266 | -1.9% | -4.6% | 101 | FAIL |
| ETHUSDT | 1.571 | +24.5% | -4.6% | 89 | PASS |
| SOLUSDT | 1.192 | +17.9% | -3.2% | 69 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R4/S4 breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above R4 AND close > 1d EMA34 AND volume > 2.0x 20-period average.
Short when price breaks below S4 AND close < 1d EMA34 AND volume > 2.0x 20-period average.
Exit when price reverts to Camarilla Pivot point (PP) or ATR-based stoploss hits.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 25-50 trades/year per symbol.
Camarilla R4/S4 levels provide stronger breakouts with less frequency but higher reliability.
1d EMA34 provides long-term trend filter that works in both bull and bear markets.
"""

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
    
    # Load 4h data for Camarilla calculation - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for 4h timeframe (using previous bar's OHLC)
    # Camarilla: PP = (H+L+C)/3, R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    
    # First bar: use current values (will be refined as more data comes)
    prev_high[0] = high_4h[0]
    prev_low[0] = low_4h[0]
    prev_close[0] = close_4h[0]
    
    camarilla_pp = (prev_high + prev_low + prev_close) / 3.0
    camarilla_range = prev_high - prev_low
    camarilla_r4 = prev_close + camarilla_range * 1.1 / 2.0
    camarilla_s4 = prev_close - camarilla_range * 1.1 / 2.0
    
    # Align 4h Camarilla levels to 4h timeframe (no additional delay needed as they're based on completed bar)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s4)
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d data
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) on 4h data for stoploss
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R4 AND close > 1d EMA34 AND volume spike
            if (price > camarilla_r4_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S4 AND close < 1d EMA34 AND volume spike
            elif (price < camarilla_s4_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to PP or ATR stoploss
                if price <= camarilla_pp_aligned[i]:
                    exit_signal = True
                elif price < entry_price - 2.5 * atr_4h[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to PP or ATR stoploss
                if price >= camarilla_pp_aligned[i]:
                    exit_signal = True
                elif price > entry_price + 2.5 * atr_4h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R4S4_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 03:55
