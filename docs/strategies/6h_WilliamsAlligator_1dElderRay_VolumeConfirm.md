# Strategy: 6h_WilliamsAlligator_1dElderRay_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.076 | +16.0% | -18.9% | 261 | FAIL |
| ETHUSDT | 0.615 | +60.8% | -10.5% | 244 | PASS |
| SOLUSDT | 0.431 | +59.6% | -30.3% | 221 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.304 | +10.4% | -10.1% | 89 | PASS |
| SOLUSDT | -1.195 | -13.5% | -20.1% | 74 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d Elder Ray + volume confirmation
# Long when: Alligator jaws < teeth < lips (bullish alignment), 1d Bull Power > 0, volume > 1.5x 24-period MA (6h equivalent)
# Short when: Alligator jaws > teeth > lips (bearish alignment), 1d Bear Power < 0, volume > 1.5x 24-period MA
# Exit when: Alligator alignment reverses (jaws-teeth-lips cross) or Elder Ray power changes sign
# Uses Alligator for trend definition, Elder Ray for bull/bear power confirmation, volume for conviction
# Timeframe: 6h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_WilliamsAlligator_1dElderRay_VolumeConfirm"
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
    open_price = prices['open'].values
    
    # Calculate volume confirmation on 6h using 24-period MA (equivalent to 1d lookback)
    if len(volume) >= 24:
        vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
        volume_filter = volume > (1.5 * vol_ma_24)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Elder Ray (standard setting)
    if len(close_1d) >= 13:
        ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
        bull_power = high_1d - ema_13_1d  # Bull Power = High - EMA13
        bear_power = low_1d - ema_13_1d   # Bear Power = Low - EMA13
    else:
        bull_power = np.full(len(close_1d), np.nan)
        bear_power = np.full(len(close_1d), np.nan)
    
    # Align Elder Ray powers to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate Williams Alligator on 6h timeframe
    # Jaw (Blue): 13-period SMMA, smoothed 8 bars ahead
    # Teeth (Red): 8-period SMMA, smoothed 5 bars ahead  
    # Lips (Green): 5-period SMMA, smoothed 3 bars ahead
    if len(close) >= 13:
        # Jaw: SMMA(13) then smoothed 8
        jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
        jaw = pd.Series(jaw_raw).rolling(window=8, min_periods=8).mean().values
        
        # Teeth: SMMA(8) then smoothed 5
        teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean().values
        teeth = pd.Series(teeth_raw).rolling(window=5, min_periods=5).mean().values
        
        # Lips: SMMA(5) then smoothed 3
        lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().values
        lips = pd.Series(lips_raw).rolling(window=3, min_periods=3).mean().values
    else:
        jaw = np.full(n, np.nan)
        teeth = np.full(n, np.nan)
        lips = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator bullish alignment + Bull Power > 0 + volume filter
            if (jaw[i] < teeth[i] and teeth[i] < lips[i] and  # Jaws < Teeth < Lips
                bull_power_aligned[i] > 0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator bearish alignment + Bear Power < 0 + volume filter
            elif (jaw[i] > teeth[i] and teeth[i] > lips[i] and  # Jaws > Teeth > Lips
                  bear_power_aligned[i] < 0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment turns bearish OR Bull Power <= 0
            if (jaw[i] >= teeth[i] or teeth[i] >= lips[i] or bull_power_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment turns bullish OR Bear Power >= 0
            if (jaw[i] <= teeth[i] or teeth[i] <= lips[i] or bear_power_aligned[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-05 09:52
