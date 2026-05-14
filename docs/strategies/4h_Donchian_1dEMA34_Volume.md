# Strategy: 4h_Donchian_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.206 | +30.6% | -17.4% | 93 | KEEP |
| ETHUSDT | 0.207 | +31.9% | -14.0% | 100 | KEEP |
| SOLUSDT | 0.747 | +117.8% | -27.6% | 92 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.438 | -3.3% | -5.4% | 22 | DISCARD |
| ETHUSDT | 0.298 | +9.9% | -9.1% | 13 | KEEP |
| SOLUSDT | 0.030 | +6.0% | -5.1% | 11 | KEEP |

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
    
    # === 4h data (primary) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (HTF for trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === 1d EMA34 (trend filter) ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 4h Donchian channel (20-period) ===
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to avoid look-ahead (use previous bar's channel)
    donch_high = np.roll(donch_high, 1)
    donch_low = np.roll(donch_low, 1)
    donch_high[0] = np.nan
    donch_low[0] = np.nan
    
    # === 4h volume ratio for confirmation ===
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = volume_4h / vol_ma_20_4h
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ratio_4h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donch_high[i]
        lower = donch_low[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_ratio = vol_ratio_4h[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower OR trend reverses (below EMA34)
            if price < lower or price < ema_trend:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper OR trend reverses (above EMA34)
            if price > upper or price > ema_trend:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Break above Donchian upper with volume, in uptrend (above EMA34)
            if price > upper and vol_ratio > 2.0 and price > ema_trend:
                signals[i] = 0.25
                position = 1
                continue
            # SHORT: Break below Donchian lower with volume, in downtrend (below EMA34)
            elif price < lower and vol_ratio > 2.0 and price < ema_trend:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-16 19:11
