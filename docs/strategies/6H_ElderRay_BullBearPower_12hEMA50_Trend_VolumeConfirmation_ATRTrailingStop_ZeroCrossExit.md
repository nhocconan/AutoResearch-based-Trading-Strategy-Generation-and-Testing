# Strategy: 6H_ElderRay_BullBearPower_12hEMA50_Trend_VolumeConfirmation_ATRTrailingStop_ZeroCrossExit

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.006 | +18.4% | -12.9% | 206 | PASS |
| ETHUSDT | 0.204 | +32.3% | -14.2% | 202 | PASS |
| SOLUSDT | 0.954 | +183.9% | -34.9% | 182 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.454 | -0.1% | -8.4% | 71 | FAIL |
| ETHUSDT | 0.343 | +12.0% | -10.9% | 69 | PASS |
| SOLUSDT | 0.340 | +12.1% | -12.9% | 67 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power + 12h EMA50 trend + volume confirmation.
Long when Bull Power > 0 AND close > 12h EMA50 AND volume > 1.5x 20-period average.
Short when Bear Power < 0 AND close < 12h EMA50 AND volume > 1.5x 20-period average.
Exit when Elder Ray crosses zero (Bull/Bear Power changes sign) or ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.25) to balance profit and fee drag. Targets 12-37 trades/year per symbol.
Elder Ray measures bull/bear power via EMA13; EMA50 filters trend; volume spike confirms conviction.
Works in both bull (strong buying) and bear (strong selling) markets.
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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Elder Ray (Bull/Bear Power) from 1d data
    # Bull Power = High - EMA13(close)
    # Bear Power = Low - EMA13(close)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA13 on 1d close
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13)  # EMA50 needs 50, vol MA needs 20, EMA13 needs 13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema50_val = ema50_12h_aligned[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND uptrend (close > EMA50) AND volume spike
            if bull_val > 0 and close[i] > ema50_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Bear Power < 0 AND downtrend (close < EMA50) AND volume spike
            elif bear_val < 0 and close[i] < ema50_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Elder Ray zero cross (Bull/Bear Power changes sign)
            if position == 1 and bull_val <= 0:
                exit_signal = True
            elif position == -1 and bear_val >= 0:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_BullBearPower_12hEMA50_Trend_VolumeConfirmation_ATRTrailingStop_ZeroCrossExit"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-23 06:44
