# Strategy: 4h_1d_camarilla_breakout_vol_filter_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.064 | +18.5% | -11.5% | 118 | DISCARD |
| ETHUSDT | 0.430 | +40.2% | -8.7% | 92 | KEEP |
| SOLUSDT | 0.799 | +86.1% | -14.7% | 91 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.110 | +7.0% | -8.7% | 38 | KEEP |
| SOLUSDT | 1.222 | +21.7% | -5.7% | 30 | KEEP |

## Code
```python
# 4h_1d_camarilla_breakout_vol_filter_v2
# Strategy: 4h Camarilla pivot breakout with volume confirmation and volatility filter (revised)
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla levels from daily pivot provide strong support/resistance.
# Breakouts aligned with volume confirmation and volatility filter capture
# sustained moves while avoiding false breakouts. Reduced trade frequency to
# improve generalization by tightening volume and volatility filters.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_vol_filter_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    rng = prev_high - prev_low
    H3 = prev_close + 1.1 * rng / 4
    L3 = prev_close - 1.1 * rng / 4
    H4 = prev_close + 1.1 * rng / 2
    L4 = prev_close - 1.1 * rng / 2
    
    # Align Camarilla levels to 4h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: ATR(20) ratio - low volatility preferred for breakouts
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_20 / np.roll(atr_20, 20)  # Current ATR vs 20 periods ago
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(vol_avg_20[i]) or np.isnan(atr_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average (tighter)
        vol_confirm = volume[i] > 2.0 * vol_avg_20[i]
        
        # Volatility filter: only trade when volatility is expanding (ATR ratio > 1.2)
        vol_filter = atr_ratio[i] > 1.2
        
        # Breakout signals using Camarilla levels
        breakout_up = high[i] > H3_aligned[i-1]
        breakdown_down = low[i] < L3_aligned[i-1]
        
        # Entry conditions
        # Long: Breakout above H3 AND volume confirmation AND volatility filter
        if breakout_up and vol_confirm and vol_filter and position != 1:
            position = 1
            signals[i] = 0.20
        # Short: Breakdown below L3 AND volume confirmation AND volatility filter
        elif breakdown_down and vol_confirm and vol_filter and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit: Opposite breakout using H4/L4 levels
        elif position == 1 and low[i] < L4_aligned[i-1]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and high[i] > H4_aligned[i-1]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals
```

## Last Updated
2026-04-11 20:33
