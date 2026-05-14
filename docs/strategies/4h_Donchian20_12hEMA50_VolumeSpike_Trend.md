# Strategy: 4h_Donchian20_12hEMA50_VolumeSpike_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.077 | +14.0% | -22.6% | 136 | FAIL |
| ETHUSDT | 0.119 | +25.3% | -13.7% | 123 | PASS |
| SOLUSDT | 0.982 | +187.0% | -24.3% | 130 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.135 | +7.4% | -9.3% | 44 | PASS |
| SOLUSDT | 0.585 | +17.5% | -10.7% | 42 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Uses discrete position sizing (0.25) to minimize fee churn. Donchian channels provide clear
# breakout levels, while 12h EMA50 filters for higher-timeframe trend alignment. Volume spike
# confirmation reduces false breakouts. Target: 20-30 trades/year per symbol, suitable for
# both bull and bear markets via trend-following logic.

name = "4h_Donchian20_12hEMA50_VolumeSpike_Trend"
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 4h data for Donchian(20) channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian(20) upper and lower bands
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # Get 4h data for volume EMA(20) for volume confirmation
    vol_4h = df_4h['volume'].values
    vol_ema_20 = pd.Series(vol_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20_aligned[i])
        
        # 12h trend: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = close[i] > ema_50_12h_aligned[i]
        bearish_trend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper + volume confirmation + bullish 12h trend
            if (close[i] > donchian_upper_aligned[i] and volume_confirmed and bullish_trend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + volume confirmation + bearish 12h trend
            elif (close[i] < donchian_lower_aligned[i] and volume_confirmed and bearish_trend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below Donchian lower OR 12h trend turns bearish
            if close[i] < donchian_lower_aligned[i] or bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above Donchian upper OR 12h trend turns bullish
            if close[i] > donchian_upper_aligned[i] or bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-04 03:47
