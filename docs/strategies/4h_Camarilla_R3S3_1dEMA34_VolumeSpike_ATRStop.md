# Strategy: 4h_Camarilla_R3S3_1dEMA34_VolumeSpike_ATRStop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.422 | +36.2% | -7.2% | 226 | PASS |
| ETHUSDT | 0.403 | +38.7% | -10.8% | 212 | PASS |
| SOLUSDT | 0.265 | +35.6% | -12.6% | 178 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.988 | -1.0% | -4.8% | 89 | FAIL |
| ETHUSDT | 1.395 | +24.8% | -5.7% | 79 | PASS |
| SOLUSDT | -0.045 | +5.3% | -6.2% | 61 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter, volume spike confirmation (2.0x 20-period average), and ATR(14) trailing stop (3.0x).
- Long: price breaks above Camarilla R3 level + price > 1d EMA34 + volume > 2.0x 20-period average volume
- Short: price breaks below Camarilla S3 level + price < 1d EMA34 + volume > 2.0x 20-period average volume
- Exit: trailing stop (3.0x ATR from extreme) OR Camarilla breakout in opposite direction
- Uses 1d EMA34 as trend filter to avoid counter-trend trades
- Volume spike reduces false breakouts
- ATR trailing stop manages risk without look-ahead
- Designed for both bull and bear markets: trend filter adapts to regime
- Target: 19-50 trades/year (75-200 total over 4 years) to minimize fee drag on 4h timeframe
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
    
    # Calculate ATR(14) for trailing stop
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla pivot levels (using previous bar's OHLC)
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    # Using previous bar's data to avoid look-ahead
    prev_close = np.concatenate([[np.nan], close[:-1]])
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Volume confirmation: > 2.0x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d EMA34 ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 34)  # Need 20 for volume MA, 14 for ATR, 34 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Camarilla breakout conditions (using current bar's close vs previous bar's levels)
        breakout_up = close[i] > camarilla_r3[i]  # Break above Camarilla R3
        breakout_down = close[i] < camarilla_s3[i]  # Break below Camarilla S3
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Camarilla breakout up + price > 1d EMA34 + volume spike
            if breakout_up and close[i] > ema_34_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
            # Short: Camarilla breakout down + price < 1d EMA34 + volume spike
            elif breakout_down and close[i] < ema_34_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
                short_extreme = low[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, high[i])
            
            # Exit conditions:
            # 1. Price reverses 3.0x ATR from long extreme (trailing stop)
            # 2. Camarilla breakout down (opposite signal)
            trailing_stop_long = close[i] < long_extreme - 3.0 * atr[i]
            breakout_down_exit = close[i] < camarilla_s3[i]
            
            if trailing_stop_long or breakout_down_exit:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, low[i])
            
            # Exit conditions:
            # 1. Price reverses 3.0x ATR from short extreme (trailing stop)
            # 2. Camarilla breakout up (opposite signal)
            trailing_stop_short = close[i] > short_extreme + 3.0 * atr[i]
            breakout_up_exit = close[i] > camarilla_r3[i]
            
            if trailing_stop_short or breakout_up_exit:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 19:24
