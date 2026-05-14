# Strategy: 4H_Donchian20_1dEMA50_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.069 | +23.2% | -12.2% | 141 | PASS |
| ETHUSDT | 0.061 | +22.5% | -17.0% | 138 | PASS |
| SOLUSDT | 0.662 | +84.7% | -21.4% | 130 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.848 | -0.7% | -4.3% | 49 | FAIL |
| ETHUSDT | 0.502 | +13.2% | -6.2% | 50 | PASS |
| SOLUSDT | 0.730 | +17.1% | -6.8% | 39 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation.
Long when price breaks above Donchian(20) high AND close > 1d EMA50 AND volume > 1.8x 20-period average.
Short when price breaks below Donchian(20) low AND close < 1d EMA50 AND volume > 1.8x 20-period average.
Exit when price crosses Donchian midpoint or ATR-based stoploss (2.0x ATR).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 15-35 trades/year per symbol.
Donchian channels provide objective breakout levels, while 1d EMA50 ensures alignment with daily trend.
Volume confirmation filters weak breakouts. Works in both bull (trend continuation) and bear (mean reversion) regimes.
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
    
    # Load 4h data for Donchian calculation - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian(20) on 4h data
    highest_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high = highest_high
    donchian_low = lowest_low
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align 4h Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
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
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND close > 1d EMA50 AND volume spike
            if (price > donchian_high_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian low AND close < 1d EMA50 AND volume spike
            elif (price < donchian_low_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below Donchian midpoint or ATR stoploss
                if price < donchian_mid_aligned[i]:
                    exit_signal = True
                elif price < entry_price - 2.0 * atr_4h[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above Donchian midpoint or ATR stoploss
                if price > donchian_mid_aligned[i]:
                    exit_signal = True
                elif price > entry_price + 2.0 * atr_4h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 04:11
