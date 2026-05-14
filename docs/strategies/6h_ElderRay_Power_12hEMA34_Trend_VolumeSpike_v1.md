# Strategy: 6h_ElderRay_Power_12hEMA34_Trend_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.303 | +30.1% | -3.9% | 376 | PASS |
| ETHUSDT | 0.019 | +21.6% | -8.4% | 310 | PASS |
| SOLUSDT | 0.004 | +19.5% | -13.4% | 270 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.531 | -3.2% | -7.0% | 140 | FAIL |
| ETHUSDT | 0.467 | +10.7% | -4.8% | 125 | PASS |
| SOLUSDT | 0.406 | +10.2% | -4.2% | 103 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h EMA34 trend filter + volume spike confirmation
# Elder Ray measures bull/bear power relative to EMA13, works in all regimes by confirming with 12h trend and volume.
# Target: 12-37 trades/year (50-150 over 4 years) to minimize fee drag

name = "6h_ElderRay_Power_12hEMA34_Trend_VolumeSpike_v1"
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
    
    # 12h HTF data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Elder Ray components: EMA13 of close
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Bull Power = High - EMA13
    bull_power = high - ema_13
    # Bear Power = Low - EMA13
    bear_power = low - ema_13
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(34, 20, 13)  # Need sufficient history for 12h EMA, volume MA, and EMA13
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_ma_20[i]) or np.isnan(ema_13[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray conditions
        bull_strong = bull_power[i] > 0  # Bulls in control
        bear_strong = bear_power[i] < 0  # Bears in control
        
        # Trend filter: price above/below 12h EMA34
        uptrend = close[i] > ema_34_12h_aligned[i]
        downtrend = close[i] < ema_34_12h_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull power positive, volume spike, uptrend
            if bull_strong and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Bear power negative, volume spike, downtrend
            elif bear_strong and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on bear power turning positive or trend reversal
            if bear_power[i] >= 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on bull power turning negative or trend reversal
            if bull_power[i] <= 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-01 19:49
