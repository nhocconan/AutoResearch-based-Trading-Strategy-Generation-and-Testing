# Strategy: 6h_Camarilla_R4S4_Breakout_12hEMA50_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.311 | +35.8% | -8.2% | 131 | PASS |
| ETHUSDT | 0.282 | +36.3% | -15.7% | 116 | PASS |
| SOLUSDT | 0.848 | +126.0% | -27.3% | 93 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.838 | -3.1% | -8.6% | 48 | FAIL |
| ETHUSDT | 1.079 | +24.9% | -8.2% | 42 | PASS |
| SOLUSDT | 0.417 | +12.8% | -11.3% | 31 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R4/S4 Breakout with 12h EMA50 Trend and Volume Spike Filter
- Camarilla R4/S4 levels represent stronger intraday support/resistance from daily extremes (breakout/continuation zones)
- 12h EMA(50) ensures alignment with higher timeframe trend for multi-timeframe confirmation
- Volume > 2.0x 20-period average confirms strong breakout momentum and reduces false signals
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years) to minimize fee drag
- Works in bull markets via breakouts with trend, in bear markets via fade of overextended moves at strong levels
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R4, S4 levels: R4 = close + (high-low)*1.1/2, S4 = close - (high-low)*1.1/2
    camarilla_r4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (completed 1d bar only)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Get 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA12h, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla breakout signals with trend filter and volume spike
        # Long: price breaks above Camarilla R4 + uptrend + volume spike
        # Short: price breaks below Camarilla S4 + downtrend + volume spike
        long_signal = (close[i] > camarilla_r4_aligned[i] and 
                      close[i] > ema_50_12h_aligned[i] and
                      volume[i] > 2.0 * vol_ma[i])
        
        short_signal = (close[i] < camarilla_s4_aligned[i] and 
                       close[i] < ema_50_12h_aligned[i] and
                       volume[i] > 2.0 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend reversal or opposite Camarilla level break
            exit_signal = False
            
            if position == 1:
                # Exit long: trend reversal or price breaks below Camarilla S4
                if (close[i] < ema_50_12h_aligned[i] or 
                    close[i] < camarilla_s4_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: trend reversal or price breaks above Camarilla R4
                if (close[i] > ema_50_12h_aligned[i] or 
                    close[i] > camarilla_r4_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-23 18:08
