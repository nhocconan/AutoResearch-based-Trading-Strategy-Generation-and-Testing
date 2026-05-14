# Strategy: 4h_Donchian_Breakout_Volume_Spike_ATR_Stop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.074 | +23.0% | -13.0% | 46 | PASS |
| ETHUSDT | -0.657 | -10.2% | -27.1% | 44 | FAIL |
| SOLUSDT | 0.331 | +48.3% | -22.2% | 47 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.159 | +7.7% | -6.1% | 17 | PASS |
| SOLUSDT | 0.445 | +14.7% | -15.4% | 11 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Spike and ATR Stop
Uses 4h Donchian channel breakouts confirmed by volume spikes and ATR-based stop loss.
Designed for low trade frequency (target: 20-50 trades/year) with strong edge in trending markets.
Works in both bull and bear markets by taking breakouts in direction of higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 4h timeframe (already aligned by get_htf_data, but ensure proper alignment)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection (2x 4-period average)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # ATR calculation for stop loss
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band with volume spike and above daily EMA
            if (price > upper and 
                volume_spike[i] and 
                price > ema_trend):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian band with volume spike and below daily EMA
            elif (price < lower and 
                  volume_spike[i] and 
                  price < ema_trend):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit conditions: stop loss or reversal signal
            if price <= entry_price - 2.0 * atr_val:  # Stop loss
                signals[i] = 0.0
                position = 0
            elif price < ema_trend:  # Trend reversal
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit conditions: stop loss or reversal signal
            if price >= entry_price + 2.0 * atr_val:  # Stop loss
                signals[i] = 0.0
                position = 0
            elif price > ema_trend:  # Trend reversal
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian_Breakout_Volume_Spike_ATR_Stop"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 00:18
