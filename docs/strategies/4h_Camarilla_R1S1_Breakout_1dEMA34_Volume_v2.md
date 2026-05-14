# Strategy: 4h_Camarilla_R1S1_Breakout_1dEMA34_Volume_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.113 | +24.7% | -7.5% | 1023 | KEEP |
| ETHUSDT | 0.036 | +22.0% | -7.7% | 990 | KEEP |
| SOLUSDT | 0.067 | +22.4% | -21.4% | 920 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.745 | -6.0% | -8.1% | 346 | DISCARD |
| ETHUSDT | 0.144 | +7.4% | -13.0% | 331 | KEEP |
| SOLUSDT | 0.878 | +16.1% | -5.7% | 332 | KEEP |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation (>1.5x 20-bar avg volume).
# Uses Camarilla pivot levels from 1d timeframe for structure, EMA34 for higher timeframe trend alignment, volume spike for participation confirmation.
# Designed for BTC/ETH with discrete sizing (0.30) to minimize fee churn while capturing strong momentum moves in both bull and bear markets.
# Target: 75-200 total trades over 4 years on 4h timeframe.

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Volume_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (based on prior 1d bar)
    # R1 = close + 1.1*(high-low)*1.05/4
    # S1 = close - 1.1*(high-low)*1.05/4
    prior_1d_high = df_1d['high'].values
    prior_1d_low = df_1d['low'].values
    prior_1d_close = df_1d['close'].values
    
    camarilla_r1 = prior_1d_close + 1.1 * (prior_1d_high - prior_1d_low) * 1.05 / 4
    camarilla_s1 = prior_1d_close - 1.1 * (prior_1d_high - prior_1d_low) * 1.05 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_vol, 1), n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R1, close > 1d EMA34, volume spike
            if (high[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.30  # Full position on breakout
                position = 1
            # SHORT: Price breaks below Camarilla S1, close < 1d EMA34, volume spike
            elif (low[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.30  # Full position on breakout
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # CONTINUE LONG: Reduce to half position if still above R1 and volume OK
            if (high[i] > camarilla_r1_aligned[i] and 
                volume[i] > avg_volume[i]):
                signals[i] = 0.15  # Half position to reduce churn
            else:
                signals[i] = 0.0  # Exit if breaks below R1 or low volume
                position = 0
        elif position == -1:
            # CONTINUE SHORT: Reduce to half position if still below S1 and volume OK
            if (low[i] < camarilla_s1_aligned[i] and 
                volume[i] > avg_volume[i]):
                signals[i] = -0.15  # Half position to reduce churn
            else:
                signals[i] = 0.0  # Exit if breaks above S1 or low volume
                position = 0
    
    return signals
```

## Last Updated
2026-05-13 22:32
