# Strategy: 4h_Donchian20_1dEMA34_Volume_Trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.205 | +30.8% | -16.8% | 108 | PASS |
| ETHUSDT | 0.116 | +25.2% | -18.0% | 113 | PASS |
| SOLUSDT | 0.726 | +116.3% | -24.6% | 108 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.375 | -8.5% | -10.6% | 50 | FAIL |
| ETHUSDT | 0.136 | +7.5% | -10.1% | 38 | PASS |
| SOLUSDT | 0.570 | +16.8% | -12.1% | 33 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA trend filter and volume confirmation
# Uses 4h timeframe for signal generation with Donchian channel breakouts
# 1d EMA(34) determines primary trend direction (bullish/bearish) - multi-timeframe alignment
# Volume confirmation (1.8x 20-period average) ensures institutional participation
# Discrete position sizing (0.25) balances return and risk while minimizing fee drag
# Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe
# Donchian provides objective price channels, volume confirms breakout validity
# 1d EMA filter ensures trades only occur in direction of higher timeframe trend
# Works in both bull and bear markets by only taking trades aligned with 1d trend

name = "4h_Donchian20_1dEMA34_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34) for trend determination
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period) on 4h
    # Upper band = 20-period high, Lower band = 20-period low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (1.8x 20-period average) - reduced from 2.0 to increase signal quality
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Close > Donchian upper + volume confirm + price > 1d EMA34 (bullish trend)
            if close[i] > donchian_upper[i] and volume_confirm[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close < Donchian lower + volume confirm + price < 1d EMA34 (bearish trend)
            elif close[i] < donchian_lower[i] and volume_confirm[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close < Donchian lower (breakdown) or price < 1d EMA34 (trend reversal)
            if close[i] < donchian_lower[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close > Donchian upper (breakout) or price > 1d EMA34 (trend reversal)
            if close[i] > donchian_upper[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 20:51
