# Strategy: 6h_1d_camarilla_breakout_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.209 | +14.5% | -10.5% | 213 | FAIL |
| ETHUSDT | 0.030 | +21.7% | -9.1% | 190 | PASS |
| SOLUSDT | 0.065 | +22.0% | -16.3% | 133 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.562 | +24.8% | -6.0% | 65 | PASS |
| SOLUSDT | 0.136 | +7.3% | -5.4% | 59 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation
    # Enter long when price breaks above R4 with volume > 2x 20-bar avg
    # Enter short when price breaks below S4 with volume > 2x 20-bar avg
    # Exit when price crosses the 1d close (midpoint)
    # Uses 1d HTF for Camarilla levels (more stable than 6h) and 6h for entry timing
    # Camarilla levels from 1d provide institutional support/resistance
    # Volume confirmation ensures breakouts have participation
    # Works in bull (continuation breaks) and bear (reversal breaks at extremes)
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for primary timeframe
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Get 1d data for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, S4 = close - 1.1*(high-low)*1.1/2
    # Actually: R4 = close + (high-low)*1.1/2, S4 = close - (high-low)*1.1/2
    cam_high_low = high_1d - low_1d
    camarilla_r4 = close_1d + (cam_high_low * 1.1 / 2)
    camarilla_s4 = close_1d - (cam_high_low * 1.1 / 2)
    camarilla_mid = close_1d  # midpoint is the 1d close
    
    # Align 1d Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    
    # Volume confirmation: volume > 2x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(1, n):  # start from 1 to access previous bar
        # Skip if data not ready
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or np.isnan(camarilla_mid_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions (using current bar's close vs current bar's levels)
        breakout_up = close[i] > camarilla_r4_aligned[i]  # break above R4
        breakout_down = close[i] < camarilla_s4_aligned[i]  # break below S4
        
        # Entry conditions with volume confirmation
        long_entry = breakout_up and volume_confirmed[i] and position != 1
        short_entry = breakout_down and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and close[i] < camarilla_mid_aligned[i])
        exit_short = (position == -1 and close[i] > camarilla_mid_aligned[i])
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-13 07:52
