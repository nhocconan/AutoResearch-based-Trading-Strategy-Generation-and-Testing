# Strategy: 6h_Camarilla_Pivot_Breakout_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.590 | +42.7% | -4.6% | 148 | PASS |
| ETHUSDT | 0.223 | +29.9% | -12.1% | 135 | PASS |
| SOLUSDT | 0.717 | +84.2% | -18.8% | 121 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.613 | +1.2% | -4.5% | 59 | FAIL |
| ETHUSDT | 1.760 | +30.5% | -5.8% | 48 | PASS |
| SOLUSDT | -0.392 | +0.9% | -7.3% | 48 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivots provide mathematically derived support/resistance levels that work in both trending and ranging markets
# Breakouts at R4/S4 levels indicate strong momentum continuation, while R3/S3 offer mean reversion opportunities
# 1d EMA34 ensures trades align with intermediate-term trend to reduce false signals
# Volume confirmation at 2.0x average filters low-participation moves
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Discrete sizing 0.25 to balance profit potential and fee drag

name = "6h_Camarilla_Pivot_Breakout_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla pivot levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Resistance levels
    r3 = pivot + range_hl * 1.1 / 4.0
    r4 = pivot + range_hl * 1.1 / 2.0
    
    # Support levels
    s3 = pivot - range_hl * 1.1 / 4.0
    s4 = pivot - range_hl * 1.1 / 2.0
    
    # Align 1d levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: Price breaks above R4 AND price > 1d EMA34 AND volume spike
            if (close[i] > r4_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: Price breaks below S4 AND price < 1d EMA34 AND volume spike
            elif (close[i] < s4_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below R3 (mean reversion) OR closes below 1d EMA34 (trend change)
            if close[i] < r3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above S3 (mean reversion) OR closes above 1d EMA34 (trend change)
            if close[i] > s3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 06:44
