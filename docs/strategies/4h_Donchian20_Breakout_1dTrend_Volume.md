# Strategy: 4h_Donchian20_Breakout_1dTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.519 | +47.6% | -8.1% | 162 | PASS |
| ETHUSDT | 0.223 | +32.4% | -10.7% | 162 | PASS |
| SOLUSDT | 0.882 | +128.3% | -21.4% | 169 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.156 | -4.7% | -7.7% | 62 | FAIL |
| ETHUSDT | 0.484 | +13.3% | -7.6% | 55 | PASS |
| SOLUSDT | 0.442 | +12.6% | -8.2% | 50 | PASS |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (EMA34) and volume confirmation (1.5x MA20).
# Enters long when price breaks above Donchian high with 1d bullish trend (close > EMA34) and volume > 1.5x MA20.
# Enters short when price breaks below Donchian low with 1d bearish trend (close < EMA34) and volume > 1.5x MA20.
# Exits when price crosses the 4h EMA20 (mean reversion).
# Uses discrete position sizing (0.25) to balance return and drawdown.
# Designed for low trade frequency (~25-50/year) by requiring confluence of breakout, trend, and volume.
# Works in both bull and bear markets: 1d trend filter ensures alignment with higher timeframe direction,
# while Donchian breakouts capture strong momentum moves and volume confirmation reduces false signals.

name = "4h_Donchian20_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 4h data for Donchian channels (based on previous 4h bar)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channels: upper, lower (based on previous 4h bar)
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    lookback = 20
    donchian_high = pd.Series(high_4h).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low_4h).rolling(window=lookback, min_periods=lookback).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    # 4h EMA20 for exit condition
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i]) or np.isnan(ema20_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high with 1d bullish trend and volume spike
            if close[i] > donchian_high_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low with 1d bearish trend and volume spike
            elif close[i] < donchian_low_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 4h EMA20 (mean reversion)
            if close[i] < ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 4h EMA20 (mean reversion)
            if close[i] > ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-13 12:34
