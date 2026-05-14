# Strategy: 6h_WilliamsAlligator_12hEMA50_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.141 | +26.6% | -13.1% | 123 | PASS |
| ETHUSDT | 0.143 | +27.2% | -17.4% | 111 | PASS |
| SOLUSDT | 0.984 | +152.0% | -23.5% | 118 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.159 | -5.3% | -10.5% | 43 | FAIL |
| ETHUSDT | 0.519 | +13.9% | -7.4% | 41 | PASS |
| SOLUSDT | -0.940 | -10.2% | -20.1% | 42 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator with 12h EMA50 trend filter and volume confirmation.
- Uses Williams Alligator (JAW=13, TEETH=8, LIPS=5) from 6h timeframe.
- Alligator sleeping (JAW, TEETH, LIPS intertwined) = no trend → avoid trading.
- Alligator awakening (lines diverging) + price outside mouth = trend continuation signal.
- Trend filter: price must be above/below 12h EMA50 to align with higher timeframe direction.
- Volume confirmation: > 2.0x 20-bar average to filter weak breakouts.
- Designed for 6h timeframe to capture medium-term trends with proper Alligator alignment.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 12-37 trades/year (50-150 total over 4 years) to stay fee-efficient.
- Williams Alligator is effective in both bull and bear markets by identifying true trend vs chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 6h timeframe
    # JAW (Blue): 13-period SMMA, shifted 8 bars forward
    # TEETH (Red): 8-period SMMA, shifted 5 bars forward  
    # LIPS (Green): 5-period SMMA, shifted 3 bars forward
    def smma(source, period):
        """Smoothed Moving Average"""
        if len(source) < period:
            return np.full_like(source, np.nan, dtype=float)
        result = np.full_like(source, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (N-1) + CLOSE) / N
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted positions
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 21)  # Need enough for Alligator shifts and EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator sleeping condition: lines are intertwined (market in chop)
        # Calculate average distance between lines
        jaw_teeth_dist = abs(jaw_shifted[i] - teeth_shifted[i])
        teeth_lips_dist = abs(teeth_shifted[i] - lips_shifted[i])
        lips_jaw_dist = abs(lips_shifted[i] - jaw_shifted[i])
        avg_dist = (jaw_teeth_dist + teeth_lips_dist + lips_jaw_dist) / 3.0
        
        # Normalize by price to get relative distance
        price_norm = close[i] if close[i] != 0 else 1.0
        rel_dist = avg_dist / price_norm
        
        # Alligator is sleeping if lines are close together (choppy market)
        sleeping = rel_dist < 0.005  # 0.5% of price
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Only trade if Alligator is awakening (not sleeping) and volume confirms
            if not sleeping and volume_confirm:
                # Long: price above all Alligator lines AND above 12h EMA50
                if (close[i] > jaw_shifted[i] and close[i] > teeth_shifted[i] and 
                    close[i] > lips_shifted[i] and close[i] > ema_50_12h_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: price below all Alligator lines AND below 12h EMA50
                elif (close[i] < jaw_shifted[i] and close[i] < teeth_shifted[i] and 
                      close[i] < lips_shifted[i] and close[i] < ema_50_12h_aligned[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price crosses below TEETH (8-period) OR below 12h EMA50
            if close[i] < teeth_shifted[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above TEETH (8-period) OR above 12h EMA50
            if close[i] > teeth_shifted[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_12hEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-24 01:10
