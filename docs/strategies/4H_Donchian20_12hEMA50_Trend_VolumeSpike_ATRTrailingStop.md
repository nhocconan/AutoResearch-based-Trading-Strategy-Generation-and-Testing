# Strategy: 4H_Donchian20_12hEMA50_Trend_VolumeSpike_ATRTrailingStop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.026 | +18.9% | -12.2% | 74 | FAIL |
| ETHUSDT | 0.118 | +25.6% | -14.4% | 68 | PASS |
| SOLUSDT | 0.570 | +78.5% | -21.7% | 71 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.064 | +23.3% | -6.3% | 23 | PASS |
| SOLUSDT | 0.143 | +7.6% | -7.3% | 23 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above 1d Donchian high AND price > 12h EMA50 AND volume > 1.8x 20-period average.
Short when price breaks below 1d Donchian low AND price < 12h EMA50 AND volume > 1.8x 20-period average.
Exit when price retraces to 1d Donchian midpoint or ATR trailing stop hit (2.0*ATR from highest/lowest since entry).
Uses discrete position sizing (0.30) for balanced risk/return.
Designed for 4h timeframe targeting ~25-40 trades/year per symbol (100-160 total over 4 years).
Focus on BTC and ETH as primary targets with SOL as secondary confirmation.
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
    
    # Calculate 1d Donchian(20) levels: upper, lower, midpoint
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Donchian channels: 20-period high/low
    donchian_high = pd.Series(h_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(l_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align 1d Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume average (20-period) on 4h timeframe
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
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        dh_val = donchian_high_aligned[i]
        dl_val = donchian_low_aligned[i]
        dm_val = donchian_mid_aligned[i]
        ema_50_val = ema_50_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 1d Donchian high AND price > 12h EMA50 AND volume spike
            if (price > dh_val and price > ema_50_val and volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.30
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: Price breaks below 1d Donchian low AND price < 12h EMA50 AND volume spike
            elif (price < dl_val and price < ema_50_val and volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.30
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
            
            # Primary exit: Price retraces to 1d Donchian midpoint
            if position == 1 and price <= dm_val:
                exit_signal = True
            elif position == -1 and price >= dm_val:
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
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4H_Donchian20_12hEMA50_Trend_VolumeSpike_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 06:29
