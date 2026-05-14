# Strategy: 6h_Camarilla_R1_S1_Breakout_Volume_1dEMA50

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.088 | +23.9% | -12.7% | 220 | PASS |
| ETHUSDT | 0.558 | +43.0% | -7.5% | 193 | PASS |
| SOLUSDT | 0.378 | +47.1% | -18.6% | 180 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.346 | +3.3% | -3.8% | 83 | FAIL |
| ETHUSDT | 0.300 | +9.3% | -7.2% | 76 | PASS |
| SOLUSDT | -0.221 | +1.9% | -14.4% | 60 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla Pivot Breakout + Volume Confirmation + 1d EMA50 Trend Filter
# Camarilla pivot levels (R1,S1) provide precise entry/exit points with high probability of reversal or continuation.
# Breakout beyond R1/S1 with volume confirmation indicates institutional interest.
# 1d EMA50 filter ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Works in bull markets (breakouts above R1) and bear markets (breakdowns below S1).
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.
name = "6h_Camarilla_R1_S1_Breakout_Volume_1dEMA50"
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
    
    # Get 1d data for Camarilla pivots and EMA50
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    # Using previous day's data to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and support/resistance levels (using previous day)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_ = high_1d - low_1d
    
    # Camarilla levels
    r1 = close_1d + (range_ * 1.1 / 12)
    s1 = close_1d - (range_ * 1.1 / 12)
    r4 = close_1d + (range_ * 1.1 / 2)
    s4 = close_1d - (range_ * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (using previous day's values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate EMA50 on 1d data for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume spike: current volume > 2.0 * 24-period average volume (4 days on 6h chart)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        ema_val = ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: Close above R1 AND price above EMA50 AND volume spike
            if close_val > r1_val and close_val > ema_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below S1 AND price below EMA50 AND volume spike
            elif close_val < s1_val and close_val < ema_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close below EMA50 (trend change) or at R4 (take profit)
            if close_val < ema_val or close_val >= r4_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close above EMA50 (trend change) or at S4 (take profit)
            if close_val > ema_val or close_val <= s4_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-18 22:11
