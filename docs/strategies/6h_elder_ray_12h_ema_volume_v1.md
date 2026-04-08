# Strategy: 6h_elder_ray_12h_ema_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.193 | +13.1% | -9.9% | 986 | FAIL |
| ETHUSDT | -0.470 | -0.4% | -20.9% | 943 | FAIL |
| SOLUSDT | 0.346 | +45.6% | -17.3% | 923 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.011 | +5.7% | -9.8% | 286 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 12h trend filter and volume confirmation
# Uses Elder Ray (Bull/Bear Power) on 6h for momentum detection:
# - Bull Power = High - EMA13, Bear Power = Low - EMA13
# - Long when Bull Power > 0 and rising, in 12h uptrend, with volume confirmation
# - Short when Bear Power < 0 and falling, in 12h downtrend, with volume confirmation
# - 12h EMA50 filter ensures trades align with higher timeframe trend
# - Volume confirmation (current volume > 20-period average) avoids false signals
# Designed for low frequency (target: 12-30 trades/year) to minimize fee drain
# Elder Ray captures institutional buying/selling pressure; works in trends and ranges

name = "6h_elder_ray_12h_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # EMA13 for Elder Ray (on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter from 12h EMA
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Elder Ray conditions
        bull_rising = bull_power[i] > bull_power[i-1] if i > 0 else False
        bear_falling = bear_power[i] < bear_power[i-1] if i > 0 else False
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit when bull power fades or trend changes
            if not bull_rising or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit when bear power fades or trend changes
            if not bear_falling or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Entry conditions with trend and volume confirmation
            # Long when bull power positive and rising in uptrend
            if bull_power[i] > 0 and bull_rising and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short when bear power negative and falling in downtrend
            elif bear_power[i] < 0 and bear_falling and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 05:41
