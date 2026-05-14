# Strategy: 4h_Donchian20_1dEMA34_VolumeSpike_2.0x

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.320 | +34.0% | -11.1% | 135 | PASS |
| ETHUSDT | 0.414 | +42.4% | -12.9% | 129 | PASS |
| SOLUSDT | 0.572 | +70.9% | -22.7% | 122 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.196 | -3.5% | -5.1% | 57 | FAIL |
| ETHUSDT | 0.830 | +18.3% | -5.1% | 45 | PASS |
| SOLUSDT | -0.088 | +4.2% | -9.5% | 39 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation (2.0x)
# Long when price breaks above 4h Donchian upper band AND price > 1d EMA34 (uptrend) AND volume > 2.0x 20-period average
# Short when price breaks below 4h Donchian lower band AND price < 1d EMA34 (downtrend) AND volume > 2.0x 20-period average
# Exit when price crosses 4h Donchian midpoint OR 1d EMA34 filter reverses
# Uses Donchian channel for structure + volume confirmation to reduce false breakouts
# 1d EMA34 provides strong trend filter for BTC/ETH in both bull and bear markets
# Designed for 75-200 total trades over 4 years (19-50/year) to minimize fee drag
# Timeframe: 4h (primary), HTF: 1d

name = "4h_Donchian20_1dEMA34_VolumeSpike_2.0x"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian(20) channels (based on previous 20 bars' OHLC)
    # Upper band = max(high of last 20 periods), Lower band = min(low of last 20 periods)
    # We use rolling window on previous bars to avoid look-ahead
    high_ma_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma_20
    donchian_lower = low_ma_20
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Get 1d data ONCE before loop for EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation on 4h (threshold: 2.0x for reduced frequency)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND price > EMA34 (uptrend) AND volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND price < EMA34 (downtrend) AND volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint OR price < EMA34 (trend weakening)
            if close[i] < donchian_mid_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint OR price > EMA34 (trend weakening)
            if close[i] > donchian_mid_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-05 15:01
