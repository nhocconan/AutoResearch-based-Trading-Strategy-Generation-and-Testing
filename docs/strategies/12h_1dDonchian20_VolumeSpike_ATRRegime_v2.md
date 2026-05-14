# Strategy: 12h_1dDonchian20_VolumeSpike_ATRRegime_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.614 | +6.7% | -5.4% | 9 | FAIL |
| ETHUSDT | 0.039 | +22.0% | -10.3% | 10 | PASS |
| SOLUSDT | 0.105 | +24.8% | -11.9% | 6 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.039 | +6.3% | -5.3% | 4 | PASS |
| SOLUSDT | -2.067 | -1.5% | -5.6% | 2 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with volume confirmation and ATR-based volatility filter
# Long when price breaks above 1d Donchian upper band AND volume > 1.5 * 20-period avg volume AND ATR(14) < ATR(50) (low vol regime)
# Short when price breaks below 1d Donchian lower band with same conditions
# Exit when price crosses 1d Donchian middle band (mean reversion to equilibrium)
# Uses discrete sizing 0.25 to limit drawdown and reduce fee churn
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Donchian provides daily structure, volume confirms participation, ATR filter avoids choppy markets

name = "12h_1dDonchian20_VolumeSpike_ATRRegime_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian channels and ATR regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian(20) channels
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_upper_1d = rolling_max(high_1d, 20)
    donchian_lower_1d = rolling_min(low_1d, 20)
    donchian_middle_1d = (donchian_upper_1d + donchian_lower_1d) / 2.0
    
    # Calculate 1d ATR for volatility regime filter
    def calculate_atr(high, low, close, period=14):
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        atr = np.full_like(tr, np.nan, dtype=float)
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_14_1d = calculate_atr(high_1d, low_1d, close_1d, period=14)
    atr_50_1d = calculate_atr(high_1d, low_1d, close_1d, period=50)
    atr_regime_1d = atr_14_1d < atr_50_1d  # Low volatility regime
    
    # Get 4h data ONCE before loop for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume_4h > (1.5 * avg_volume_20_4h)
    
    # Align 1d indicators to 12h timeframe (wait for completed 1d bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle_1d)
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime_1d)
    
    # Align 4h indicators to 12h timeframe (wait for completed 4h bar)
    volume_spike_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(atr_regime_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper with volume spike and low vol regime
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                volume_spike_aligned[i] and atr_regime_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower with volume spike and low vol regime
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  volume_spike_aligned[i] and atr_regime_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d Donchian middle (mean reversion)
            if close[i] < donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d Donchian middle (mean reversion)
            if close[i] > donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-06 08:30
