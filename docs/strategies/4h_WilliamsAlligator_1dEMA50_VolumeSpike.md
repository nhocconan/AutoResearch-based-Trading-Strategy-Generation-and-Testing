# Strategy: 4h_WilliamsAlligator_1dEMA50_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.015 | +21.0% | -10.3% | 107 | PASS |
| ETHUSDT | 0.027 | +20.9% | -13.8% | 90 | PASS |
| SOLUSDT | 0.863 | +113.8% | -21.8% | 86 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.252 | +3.9% | -3.9% | 36 | FAIL |
| ETHUSDT | 0.631 | +14.9% | -6.2% | 30 | PASS |
| SOLUSDT | -0.528 | -2.5% | -17.1% | 30 | FAIL |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation.
# Uses Williams Alligator (Jaw/Teeth/Lips) to identify trend direction and strength.
# Combines with 1d EMA(50) trend filter and volume spike (2x 20-period average) for confirmation.
# Aims to capture strong trends while avoiding choppy markets. Designed for 4h timeframe
# with ~50-150 total trades over 4 years to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: SMAs with specific offsets
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    def smma(data, period):
        """Smoothed Moving Average"""
        sma = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            sma[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply shifts (Jaw: +8, Teeth: +5, Lips: +3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Volatility filter: ATR(20) for dynamic thresholds
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Volume filter: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 13)  # Wait for EMA, ATR, and Alligator
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(atr[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA(50)
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, reverse = downtrend
        alligator_long = lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i]
        alligator_short = lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i]
        
        # Volatility-based entry filter: only trade when volatility is elevated
        vol_filter = atr[i] > np.mean(atr[max(0, i-50):i+1])  # Above average volatility
        
        # Entry conditions
        long_entry = uptrend and alligator_long and volume_spike[i] and vol_filter
        short_entry = downtrend and alligator_short and volume_spike[i] and vol_filter
        
        # Exit conditions: trend reversal or Alligator reversal
        long_exit = (not uptrend) or (not alligator_long)
        short_exit = (not downtrend) or (not alligator_short)
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-28 09:19
