# Strategy: 6h_ElderRay_1dADX_Regime_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.098 | +24.5% | -10.4% | 224 | PASS |
| ETHUSDT | 0.547 | +56.2% | -12.4% | 185 | PASS |
| SOLUSDT | 0.944 | +138.3% | -18.1% | 156 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.999 | -13.3% | -14.9% | 91 | FAIL |
| ETHUSDT | 0.307 | +10.5% | -8.6% | 69 | PASS |
| SOLUSDT | -1.314 | -14.3% | -23.9% | 64 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX Regime Filter and Volume Confirmation
# Long when Bull Power > 0 (close > EMA13) AND ADX > 25 (trending) AND volume spike
# Short when Bear Power < 0 (close < EMA13) AND ADX > 25 (trending) AND volume spike
# Elder Ray measures trend strength via price-EMA deviation; ADX filters for trending markets only
# Volume spike (2.0x 20-bar MA) confirms institutional participation
# Works in bull (strong uptrend + buying pressure) and bear (strong downtrend + selling pressure)
# Timeframe: 6h (primary timeframe as required)
# Target: 60-120 total trades over 4 years (15-30/year) to balance signal quality and fee drag

name = "6h_ElderRay_1dADX_Regime_VolumeSpike"
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
    
    # Get 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value: simple average
        result[period-1] = np.nansum(values[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, (dm_plus_smooth / atr_1d) * 100, 0)
    di_minus = np.where(atr_1d != 0, (dm_minus_smooth / atr_1d) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0,
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1d = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h EMA13 for Elder Ray
    if len(close) >= 13:
        ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    else:
        ema_13 = np.full(n, np.nan)
    
    # Elder Ray: Bull Power = Close - EMA13, Bear Power = EMA13 - Close
    bull_power = close - ema_13
    bear_power = ema_13 - close
    
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
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND ADX > 25 (trending) AND volume spike
            if (bull_power[i] > 0 and 
                adx_1d_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 AND ADX > 25 (trending) AND volume spike
            elif (bear_power[i] > 0 and 
                  adx_1d_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR ADX <= 25 (trend weakening)
            if bull_power[i] <= 0 or adx_1d_aligned[i] <= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power <= 0 OR ADX <= 25 (trend weakening)
            if bear_power[i] <= 0 or adx_1d_aligned[i] <= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-05 13:50
