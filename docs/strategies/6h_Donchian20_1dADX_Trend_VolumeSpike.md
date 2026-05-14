# Strategy: 6h_Donchian20_1dADX_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.247 | +9.8% | -10.0% | 120 | FAIL |
| ETHUSDT | 0.407 | +43.6% | -10.6% | 102 | PASS |
| SOLUSDT | 0.760 | +103.5% | -22.6% | 87 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.434 | +12.4% | -7.9% | 37 | PASS |
| SOLUSDT | -0.221 | +1.9% | -12.5% | 30 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d ADX trend filter + volume confirmation
# Long when price breaks above 6h Donchian upper AND 1d ADX > 25 (strong trend) AND volume spike
# Short when price breaks below 6h Donchian lower AND 1d ADX > 25 (strong trend) AND volume spike
# Exit when price crosses the 6h Donchian middle (mean) OR ADX < 20 (trend weakening)
# Uses Donchian channels for structure, 1d ADX for regime filtering (avoid whipsaws in ranging markets)
# Volume spike confirms institutional participation at breakouts
# Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# Timeframe: 6h (primary timeframe as required)
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "6h_Donchian20_1dADX_Trend_VolumeSpike"
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
    
    # Get 6h data ONCE before loop for Donchian channels
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate 6h Donchian(20) channels
    # Upper = max(high, 20)
    # Lower = min(low, 20)
    # Middle = (upper + lower) / 2
    high_ma_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma_20
    donchian_lower = low_ma_20
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Get 1d data ONCE before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    # +DI = 100 * EWM(+DM, 14) / EWM(TR, 14)
    # -DI = 100 * EWM(-DM, 14) / EWM(TR, 14)
    # DX = 100 * abs(+DI - -DI) / (+DI + -DI)
    # ADX = EWM(DX, 14)
    high_shift = np.roll(high_1d, 1)
    low_shift = np.roll(low_1d, 1)
    close_shift = np.roll(close_1d, 1)
    high_shift[0] = high_1d[0]
    low_shift[0] = low_1d[0]
    close_shift[0] = close_1d[0]
    
    plus_dm = np.where((high_1d - high_shift) > (low_shift - low_1d), np.maximum(high_1d - high_shift, 0), 0)
    minus_dm = np.where((low_shift - low_1d) > (high_1d - high_shift), np.maximum(low_shift - low_1d, 0), 0)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_shift)
    tr3 = np.abs(low_1d - close_shift)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (alpha = 1/period)
    def WilderSmoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    plus_di_14 = 100 * WilderSmoothing(plus_dm, 14) / WilderSmoothing(tr, 14)
    minus_di_14 = 100 * WilderSmoothing(minus_dm, 14) / WilderSmoothing(tr, 14)
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx = WilderSmoothing(dx, 14)
    
    # Align HTF indicators to 6h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_6h, donchian_middle)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation on 6h (threshold: 2.0x)
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
            np.isnan(donchian_middle_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper AND ADX > 25 (strong uptrend) AND volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                adx_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND ADX > 25 (strong downtrend) AND volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian middle OR ADX < 20 (trend weakening)
            if close[i] < donchian_middle_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian middle OR ADX < 20 (trend weakening)
            if close[i] > donchian_middle_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-05 14:16
