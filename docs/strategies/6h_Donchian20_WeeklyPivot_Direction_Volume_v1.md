# Strategy: 6h_Donchian20_WeeklyPivot_Direction_Volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.524 | -9.6% | -24.4% | 82 | FAIL |
| ETHUSDT | 0.181 | +30.3% | -16.2% | 80 | PASS |
| SOLUSDT | 1.043 | +207.5% | -26.5% | 70 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.134 | +7.4% | -11.3% | 27 | PASS |
| SOLUSDT | -0.212 | -0.5% | -19.3% | 27 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction filter and volume confirmation
# Uses weekly pivot points from 1d data to determine structural bias (long above weekly pivot, short below)
# Donchian breakout provides entry timing in direction of weekly pivot bias
# Volume confirmation > 1.5x 20-period EMA ensures institutional participation
# Designed for low trade frequency: ~12-25 trades/year per symbol with 0.25 sizing
# Weekly pivot acts as dynamic support/resistance that works in both bull and bear markets
# Breakouts in direction of weekly pivot bias have higher follow-through probability

name = "6h_Donchian20_WeeklyPivot_Direction_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate weekly pivot points from 1d OHLC
    # Weekly pivot = (Prior week HIGH + LOW + CLOSE) / 3
    # We use the prior completed week's values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly OHLC from daily data
    # Group by week (starting Monday) - using 7-day periods for simplicity
    # Weekly high = max of prior 7 daily highs
    # Weekly low = min of prior 7 daily lows  
    # Weekly close = close of 7th prior day
    weekly_high = pd.Series(high_1d).rolling(window=7, min_periods=7).max().shift(7).values
    weekly_low = pd.Series(low_1d).rolling(window=7, min_periods=7).min().shift(7).values
    weekly_close = pd.Series(close_1d).rolling(window=7, min_periods=7).apply(lambda x: x[-1], raw=True).shift(7).values
    
    # Weekly pivot point: (HIGH + LOW + CLOSE) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # 6h Donchian(20) breakout
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().shift(1).values
    
    # Volume confirmation: volume > 1.5 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 1d data for weekly pivot (7*2=14 days min) + Donchian20 + volume EMA20
    start_idx = max(14*4, donchian_window, 20)  # 14 days * 4 (6h bars per day) = 56
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine bias from weekly pivot: long above pivot, short below pivot
        bullish_bias = close[i] > weekly_pivot_aligned[i]
        bearish_bias = close[i] < weekly_pivot_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: Donchian breakout above upper band with volume spike
                if close[i] > donchian_high[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: Donchian breakdown below lower band with volume spike
                if close[i] < donchian_low[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around pivot
        
        elif position == 1:  # Long position
            # Exit: Donchian breakdown below lower band (failure of breakout)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Donchian breakout above upper band (failure of breakdown)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-01 22:29
