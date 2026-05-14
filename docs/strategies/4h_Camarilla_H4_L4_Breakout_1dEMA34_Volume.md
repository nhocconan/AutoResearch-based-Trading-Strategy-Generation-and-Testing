# Strategy: 4h_Camarilla_H4_L4_Breakout_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.512 | +39.8% | -7.5% | 233 | KEEP |
| ETHUSDT | 0.027 | +21.4% | -11.1% | 222 | KEEP |
| SOLUSDT | 0.486 | +54.0% | -12.3% | 195 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.041 | -1.5% | -5.7% | 93 | DISCARD |
| ETHUSDT | 1.894 | +33.3% | -5.3% | 75 | KEEP |
| SOLUSDT | 1.138 | +20.1% | -5.1% | 62 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla H4/L4 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses 1d HTF for EMA34 to capture daily trend and reduce false breakouts in choppy markets.
# Camarilla H4/L4 from 4h provides proven intraday reversal/continuation levels with good historical performance.
# Volume confirmation at 2.0x average ensures strong participation while limiting trades (~20-40/year).
# Discrete sizing 0.25 to minimize fee churn. Works in bull/bear: trend filter ensures trades only with momentum.
# Target: 50-120 total trades over 4 years (12-30/year) to balance opportunity and fee drag.

name = "4h_Camarilla_H4_L4_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels H4 and L4 from 4h timeframe (using prior completed 4h bar)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Prior 4h bar's high, low, close for Camarilla calculation
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    prev_close_4h = df_4h['close'].shift(1).values
    
    # Camarilla H4 and L4 levels (proven breakout/continuation levels)
    camarilla_h4_4h = prev_close_4h + (prev_high_4h - prev_low_4h) * 1.1 / 2
    camarilla_l4_4h = prev_close_4h - (prev_high_4h - prev_low_4h) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (they are already 4h, but align for safety)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4_4h)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4_4h)
    
    # 1d EMA34 for trend filter (daily trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 2.0x 20-period average (strict threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above H4 AND price > 1d EMA34 AND volume spike
            if (close[i] > camarilla_h4_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below L4 AND price < 1d EMA34 AND volume spike
            elif (close[i] < camarilla_l4_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below L4 OR price < 1d EMA34
            if close[i] < camarilla_l4_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above H4 OR price > 1d EMA34
            if close[i] > camarilla_h4_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 06:12
