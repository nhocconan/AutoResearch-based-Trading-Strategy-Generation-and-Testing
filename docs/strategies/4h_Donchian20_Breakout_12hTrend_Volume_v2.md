# Strategy: 4h_Donchian20_Breakout_12hTrend_Volume_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.040 | +21.6% | -9.8% | 160 | PASS |
| ETHUSDT | 0.760 | +73.7% | -10.3% | 144 | PASS |
| SOLUSDT | 0.693 | +92.7% | -25.9% | 142 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.330 | -6.4% | -10.0% | 63 | FAIL |
| ETHUSDT | 0.114 | +7.1% | -8.0% | 59 | PASS |
| SOLUSDT | 0.115 | +7.1% | -12.5% | 47 | PASS |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h trend filter (HMA21) and volume confirmation (2.0x MA20).
# Enters long when price breaks above Donchian high with 12h bullish trend and volume > 2.0x MA20.
# Enters short when price breaks below Donchian low with 12h bearish trend and volume > 2.0x MA20.
# Exits when price crosses the 4h EMA20 (mean reversion).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~20-40/year) by requiring strict confluence.
# Works in both bull and bear markets: 12h trend filter ensures alignment with higher timeframe direction,
# while Donchian breakouts capture strong momentum moves and volume confirmation reduces false signals.

name = "4h_Donchian20_Breakout_12hTrend_Volume_v2"
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
    lookback = 20
    donchian_high = pd.Series(high_4h).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low_4h).rolling(window=lookback, min_periods=lookback).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 12h data for trend filter (HMA21)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    # Calculate HMA(21) on 12h close
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_12h).ewm(span=half_len, adjust=False, min_periods=half_len).mean().values
    wma_full = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_21 = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
    hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21)
    
    # Volume filter: current volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)
    
    # 4h EMA20 for exit condition
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(hma_21_aligned[i]) or np.isnan(vol_ma20[i]) or np.isnan(ema20_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high with 12h bullish trend and volume spike
            if close[i] > donchian_high_aligned[i] and close[i] > hma_21_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low with 12h bearish trend and volume spike
            elif close[i] < donchian_low_aligned[i] and close[i] < hma_21_aligned[i] and volume_spike[i]:
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
2026-05-13 12:36
