# Strategy: 4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.412 | +43.5% | -9.9% | 102 | PASS |
| ETHUSDT | 0.453 | +52.0% | -14.6% | 93 | PASS |
| SOLUSDT | 0.852 | +126.5% | -17.5% | 80 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.626 | -0.9% | -6.8% | 33 | FAIL |
| ETHUSDT | 0.340 | +11.4% | -10.0% | 33 | PASS |
| SOLUSDT | -0.697 | -7.0% | -17.8% | 30 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation
# Camarilla pivots identify key intraday support/resistance levels where price often reverses or breaks out
# R1/S1 levels represent strong reversal zones; breaks above R1 or below S1 with volume indicate institutional participation
# 12h EMA50 provides intermediate trend filter to align with dominant momentum and reduce counter-trend whipsaws
# Volume spike (2.0x 20-period average) confirms breakout conviction
# Targets 75-200 trades over 4 years (19-50/year) for 4h timeframe
# Works in both bull and bear markets by capturing breakouts in direction of higher timeframe trend

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # Camarilla: R1 = close + 1.125*(high-low), S1 = close - 1.125*(high-low)
    # We use the previous completed 12h bar to avoid look-ahead
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    camarilla_high = close_12h + 1.125 * (high_12h - low_12h)  # R1 level
    camarilla_low = close_12h - 1.125 * (high_12h - low_12h)   # S1 level
    
    camarilla_high_aligned = align_htf_to_ltf(prices, df_12h, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_12h, camarilla_low)
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for volume MA and HTF data alignment)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Camarilla R1 + price > 12h EMA50 + volume spike
            if close[i] > camarilla_high_aligned[i] and close[i] > ema_50_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S1 + price < 12h EMA50 + volume spike
            elif close[i] < camarilla_low_aligned[i] and close[i] < ema_50_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Camarilla S1 (reversal signal)
            if close[i] < camarilla_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Camarilla R1 (reversal signal)
            if close[i] > camarilla_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 15:41
