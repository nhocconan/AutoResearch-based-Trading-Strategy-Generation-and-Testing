# Strategy: 4H_DonchianBreakout_12h_VolumeConfirmation_ATRStop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.166 | +28.7% | -15.7% | 93 | PASS |
| ETHUSDT | 0.277 | +38.8% | -13.6% | 89 | PASS |
| SOLUSDT | 1.053 | +228.7% | -27.5% | 83 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.156 | -7.4% | -12.5% | 35 | FAIL |
| ETHUSDT | 0.051 | +5.8% | -11.5% | 31 | PASS |
| SOLUSDT | 0.516 | +16.2% | -15.4% | 23 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 12h Donchian breakout with volume confirmation and ATR stoploss.
Long when price breaks above 12h Donchian upper (20) AND volume > 1.8x 20-period average.
Short when price breaks below 12h Donchian lower (20) AND volume > 1.8x 20-period average.
Exit when price retraces to 12h Donchian midpoint or ATR stoploss hit (2.0*ATR).
Uses discrete position sizing (0.30) to balance return and drawdown.
Designed for 4h timeframe to target 19-50 trades/year per symbol (75-200 total over 4 years).
Works in both bull and bear markets by using volume confirmation to filter false breakouts and ATR stops to manage risk.
12h Donchian levels provide stronger institutional support/resistance from higher timeframe.
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
    
    # Calculate 12h Donchian channels (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian channels (20-period)
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Align Donchian levels to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid)
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        mid = donchian_mid_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 12h Donchian upper AND volume spike
            if (price > upper and volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.30
                position = 1
                entry_price = price
            # Short: Price breaks below 12h Donchian lower AND volume spike
            elif (price < lower and volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.30
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to 12h Donchian midpoint
            if position == 1 and price <= mid:
                exit_signal = True
            elif position == -1 and price >= mid:
                exit_signal = True
            
            # ATR-based stoploss: 2.0 * ATR from entry
            if position == 1 and price < entry_price - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4H_DonchianBreakout_12h_VolumeConfirmation_ATRStop"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 05:56
