# Strategy: 4h_Donchian_20_12hEMA50_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.064 | +14.5% | -21.9% | 138 | DISCARD |
| ETHUSDT | 0.088 | +22.9% | -14.5% | 132 | KEEP |
| SOLUSDT | 0.825 | +143.4% | -31.3% | 138 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.022 | +5.3% | -9.8% | 44 | KEEP |
| SOLUSDT | 0.270 | +10.4% | -14.6% | 47 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation
# Uses Donchian(20) breakouts for trend capture, 12h EMA50 for trend filter, and volume spike for confirmation
# Designed to work in both bull and bear markets by using breakouts with volume confirmation and trend alignment
# Target: 20-50 trades/year to minimize fee drag while capturing significant moves

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (called ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 12h data for trend filter (called ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 4-period Donchian channels (20-period lookback)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_high = np.full(len(df_4h), np.nan)
    donchian_low = np.full(len(df_4h), np.nan)
    
    for i in range(19, len(df_4h)):
        donchian_high[i] = np.max(high_4h[i-19:i+1])
        donchian_low[i] = np.min(low_4h[i-19:i+1])
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        multiplier = 2 / (50 + 1)
        ema_50_12h[49] = np.mean(close_12h[:50])
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = (close_12h[i] * multiplier) + (ema_50_12h[i-1] * (1 - multiplier))
    
    # Calculate 4-period average volume for spike detection
    vol_ma_4h = np.full(len(df_4h), np.nan)
    vol_period = 4
    for i in range(vol_period, len(df_4h)):
        vol_ma_4h[i] = np.mean(df_4h['volume'].values[i-vol_period:i])
    
    # Align all indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(20, 50) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_4h_aligned[i] if vol_ma_4h_aligned[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume and 12h uptrend
            if price > donchian_high_aligned[i] and price > ema_50_12h_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: Price breaks below Donchian low with volume and 12h downtrend
            elif price < donchian_low_aligned[i] and price < ema_50_12h_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price breaks below Donchian low or trend reverses
            if price < donchian_low_aligned[i] or price < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price breaks above Donchian high or trend reverses
            if price > donchian_high_aligned[i] or price > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_20_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-05-08 00:36
