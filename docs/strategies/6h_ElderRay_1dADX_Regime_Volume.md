# Strategy: 6h_ElderRay_1dADX_Regime_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.008 | +18.0% | -11.7% | 243 | FAIL |
| ETHUSDT | 0.092 | +23.2% | -19.2% | 248 | PASS |
| SOLUSDT | 1.218 | +265.5% | -35.4% | 203 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.062 | +5.9% | -11.5% | 82 | PASS |
| SOLUSDT | -0.350 | -2.8% | -20.6% | 78 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter
# Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# 1d ADX > 25 indicates strong trend (trend-following mode), ADX < 20 indicates range (mean-reversion mode)
# In trend regime (ADX > 25): go long when Bull Power > 0 and rising, short when Bear Power < 0 and falling
# In range regime (ADX < 20): fade extremes - short when Bull Power > 0.5*ATR, long when Bear Power < -0.5*ATR
# Volume confirmation (>1.5 x 20-period EMA) filters low-quality signals
# Designed for 6h timeframe targeting 50-150 total trades over 4 years (12-37/year).

name = "6h_ElderRay_1dADX_Regime_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    tr_smooth = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, (dm_plus_smooth / tr_smooth) * 100, 0)
    di_minus = np.where(tr_smooth != 0, (dm_minus_smooth / tr_smooth) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, period)
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 6h ATR for dynamic thresholds
    tr_6h1 = np.abs(high[1:] - low[1:])
    tr_6h2 = np.abs(high[1:] - close[:-1])
    tr_6h3 = np.abs(low[1:] - close[:-1])
    tr_6h = np.maximum(np.maximum(tr_6h1, tr_6h2), tr_6h3)
    tr_6h = np.concatenate([[np.nan], tr_6h])
    atr_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr_6h[i]) or np.isnan(vol_ema_20[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        # Regime filter: 1d ADX > 25 = trending, ADX < 20 = ranging
        trending_regime = adx_aligned[i] > 25
        ranging_regime = adx_aligned[i] < 20
        
        if position == 0:
            if trending_regime and volume_confirm:
                # Trend-following: long when Bull Power > 0 and rising, short when Bear Power < 0 and falling
                if bull_power[i] > 0 and bull_power[i] > bull_power[i-1]:
                    signals[i] = 0.25
                    position = 1
                elif bear_power[i] < 0 and bear_power[i] < bear_power[i-1]:
                    signals[i] = -0.25
                    position = -1
            elif ranging_regime and volume_confirm:
                # Mean-reversion: fade extremes
                if bull_power[i] > 0.5 * atr_6h[i]:
                    signals[i] = -0.25  # short at bullish extreme
                    position = -1
                elif bear_power[i] < -0.5 * atr_6h[i]:
                    signals[i] = 0.25   # long at bearish extreme
                    position = 1
        elif position == 1:
            # Exit long: Bull Power turns negative OR regime shifts to ranging
            if bull_power[i] <= 0 or (not trending_regime and adx_aligned[i] < 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power turns positive OR regime shifts to ranging
            if bear_power[i] >= 0 or (not trending_regime and adx_aligned[i] < 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-04 04:08
