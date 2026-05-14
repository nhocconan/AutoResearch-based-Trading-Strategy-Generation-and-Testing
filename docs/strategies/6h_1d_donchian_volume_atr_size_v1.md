# Strategy: 6h_1d_donchian_volume_atr_size_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.496 | -17.2% | -32.3% | 106 | FAIL |
| ETHUSDT | 0.425 | +59.8% | -22.6% | 97 | PASS |
| SOLUSDT | 0.995 | +254.4% | -35.6% | 88 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.098 | +6.0% | -15.8% | 33 | PASS |
| SOLUSDT | -0.092 | +0.3% | -28.4% | 31 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1d volume spike filter and ATR-based position sizing.
    # Donchian breakouts capture momentum in both bull and bear markets.
    # Volume spike (current volume > 1.5 * 20-period MA) confirms breakout validity.
    # Position size scales with volatility: size = 0.25 * (ATR(14) / price) normalized to max 0.30.
    # Uses discrete position sizes (0.0, ±0.25) to minimize fee churn.
    # Target: 75-200 total trades over 4 years (19-50/year).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility scaling
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range components
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_1d = np.zeros_like(tr)
    atr_1d[13] = np.mean(tr[1:14])  # Seed with simple average
    for i in range(14, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align 1d ATR to 6h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume MA(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 * 20-period MA
        volume_filter = volume[i] > 1.5 * volume_ma[i]
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above prior period's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below prior period's low
        
        # Dynamic position size based on volatility (ATR/price ratio)
        # Normalize ATR ratio to target size 0.25, max 0.30
        atr_ratio = atr_1d_aligned[i] / close[i]
        base_size = 0.25
        vol_scaled_size = base_size * (atr_ratio / 0.01)  # Normalize to 1% ATR/price
        position_size = np.clip(vol_scaled_size, 0.0, 0.30)
        
        # Entry conditions: breakout with volume confirmation
        long_entry = long_breakout and volume_filter
        short_entry = short_breakout and volume_filter
        
        # Exit conditions: opposite breakout
        long_exit = short_breakout
        short_exit = long_breakout
        
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_donchian_volume_atr_size_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-13 05:38
