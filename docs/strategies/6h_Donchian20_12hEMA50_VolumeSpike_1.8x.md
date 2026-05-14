# Strategy: 6h_Donchian20_12hEMA50_VolumeSpike_1.8x

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.051 | +22.4% | -11.2% | 123 | PASS |
| ETHUSDT | 0.424 | +45.4% | -8.6% | 112 | PASS |
| SOLUSDT | 1.066 | +165.6% | -24.8% | 95 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.094 | -4.4% | -7.1% | 49 | FAIL |
| ETHUSDT | 0.579 | +14.7% | -8.1% | 36 | PASS |
| SOLUSDT | 0.216 | +8.7% | -8.0% | 31 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation (1.8x)
# Long when price breaks above 6h Donchian upper band AND price > 12h EMA50 (uptrend) AND volume > 1.8x 20-period average
# Short when price breaks below 6h Donchian lower band AND price < 12h EMA50 (downtrend) AND volume > 1.8x 20-period average
# Exit when price crosses 6h Donchian midpoint OR 12h EMA50 filter reverses
# Uses Donchian channel for structure + volume confirmation to reduce false breakouts
# 12h EMA50 provides strong trend filter for BTC/ETH in both bull and bear markets
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Timeframe: 6h (primary)

name = "6h_Donchian20_12hEMA50_VolumeSpike_1.8x"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data ONCE before loop for Donchian calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate 6h Donchian(20) channels (based on previous 20 bars' OHLC)
    # Upper band = max(high of last 20 periods), Lower band = min(low of last 20 periods)
    # We use rolling window on previous bars to avoid look-ahead
    high_ma_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma_20
    donchian_lower = low_ma_20
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Get 12h data ONCE before loop for EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50)
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_6h, donchian_mid)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation on 6h (threshold: 1.8x for balanced frequency)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.8 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND price > EMA50 (uptrend) AND volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND price < EMA50 (downtrend) AND volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint OR price < EMA50 (trend weakening)
            if close[i] < donchian_mid_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint OR price > EMA50 (trend weakening)
            if close[i] > donchian_mid_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-05 14:49
