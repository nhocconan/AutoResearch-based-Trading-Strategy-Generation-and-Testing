# Strategy: 6H_Donchian20_1dEMA50_Trend_VolumeConfirmation_ATRTrailingStop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.265 | +32.4% | -11.4% | 86 | PASS |
| ETHUSDT | 0.061 | +22.2% | -14.7% | 86 | PASS |
| SOLUSDT | 0.860 | +122.8% | -22.2% | 79 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.580 | +0.5% | -5.2% | 30 | FAIL |
| ETHUSDT | 0.066 | +6.4% | -9.0% | 28 | PASS |
| SOLUSDT | -0.293 | +1.0% | -11.4% | 30 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 12h Donchian channel breakout with 1d EMA trend filter and volume confirmation.
Long when price breaks above 12h Donchian upper channel (20-period) AND price > 1d EMA50 AND volume > 1.5x 20-period average.
Short when price breaks below 12h Donchian lower channel (20-period) AND price < 1d EMA50 AND volume > 1.5x 20-period average.
Exit when price retraces to the 12h Donchian midpoint or ATR trailing stop hit (2.0*ATR from highest/lowest since entry).
Uses discrete position sizing (0.25) to control drawdown and fee churn.
Designed for 6h timeframe to target 12-37 trades/year per symbol (50-150 total over 4 years).
Combines structure (Donchian), trend (EMA), and momentum (volume) for robustness in both bull and bear markets.
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
    
    # Calculate 12h Donchian channel (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Donchian levels: highest high and lowest low over 20 periods
    donch_high = pd.Series(df_12h['high'].values).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_12h['low'].values).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Align 12h Donchian levels to 6h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_12h, donch_mid)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume average (20-period) on 6h timeframe
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
    entry_price = 0.0
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20)  # Donchian needs 20, EMA needs 50, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or np.isnan(donch_mid_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        donch_mid_val = donch_mid_aligned[i]
        ema_50_val = ema_50_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 12h Donchian upper channel AND price > 1d EMA50 AND volume spike
            if (price > donch_high_val and price > ema_50_val and volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: Price breaks below 12h Donchian lower channel AND price < 1d EMA50 AND volume spike
            elif (price < donch_low_val and price < ema_50_val and volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to 12h Donchian midpoint
            if position == 1 and price <= donch_mid_val:
                exit_signal = True
            elif position == -1 and price >= donch_mid_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_1dEMA50_Trend_VolumeConfirmation_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-23 06:12
