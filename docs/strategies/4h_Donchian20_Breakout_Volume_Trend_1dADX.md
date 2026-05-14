# Strategy: 4h_Donchian20_Breakout_Volume_Trend_1dADX

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.381 | +0.3% | -16.7% | 186 | DISCARD |
| ETHUSDT | 0.753 | +82.0% | -16.2% | 163 | KEEP |
| SOLUSDT | 0.662 | +101.0% | -31.6% | 153 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.065 | +6.1% | -14.0% | 63 | KEEP |
| SOLUSDT | 0.594 | +17.4% | -13.7% | 51 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_Breakout_Volume_Trend_1dADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    """
    4h Donchian(20) breakout with 1d ADX trend filter and volume confirmation.
    - Long: Price breaks above Donchian(20) high with ADX>25 and volume spike
    - Short: Price breaks below Donchian(20) low with ADX>25 and volume spike
    - Exit: Price crosses back through Donchian(20) midpoint
    - Volume spike: current volume > 2.0 x 20-period average
    - Target: 20-50 trades/year on 4h timeframe
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian(20) on 4h data
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    # Calculate ADX(14) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth TR, +DM, -DM with Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    atr = np.full(len(tr), np.nan)
    plus_dm_smooth = np.full(len(tr), np.nan)
    minus_dm_smooth = np.full(len(tr), np.nan)
    
    # Initialize first value with simple average
    if len(tr) >= period:
        atr[period-1] = np.nanmean(tr[1:period])
        plus_dm_smooth[period-1] = np.nanmean(plus_dm[1:period])
        minus_dm_smooth[period-1] = np.nanmean(minus_dm[1:period])
        
        # Wilder smoothing
        for i in range(period, len(tr)):
            atr[i] = atr[i-1] - (atr[i-1] / period) + tr[i]
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / period) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / period) + minus_dm[i]
    
    # Calculate DI and DX
    plus_di = np.full(len(tr), np.nan)
    minus_di = np.full(len(tr), np.nan)
    dx = np.full(len(tr), np.nan)
    
    for i in range(period, len(tr)):
        if atr[i] > 0:
            plus_di[i] = 100 * (plus_dm_smooth[i] / atr[i])
            minus_di[i] = 100 * (minus_dm_smooth[i] / atr[i])
            if plus_di[i] + minus_di[i] > 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # ADX is smoothed DX
    adx = np.full(len(tr), np.nan)
    if len(dx) >= 2*period-1:
        adx[2*period-2] = np.nanmean(dx[period:2*period-1])
        for i in range(2*period-1, len(dx)):
            adx[i] = adx[i-1] - (adx[i-1] / period) + dx[i]
    
    # Align ADX to 4h
    adx_4h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike detection (20-period for 4h)
    vol_avg = np.full(n, np.nan)
    for i in range(20, n):
        vol_avg[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 2*period)  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_4h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian midpoint
        midpoint = (highest_high[i] + lowest_low[i]) / 2
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        # ADX trend filter: only trade when trending (ADX > 25)
        trending = adx_4h[i] > 25
        
        if position == 0:
            # Long: Break above Donchian high with trend and volume spike
            if (close[i] > highest_high[i] and trending and vol_spike):
                signals[i] = 0.30
                position = 1
            # Short: Break below Donchian low with trend and volume spike
            elif (close[i] < lowest_low[i] and trending and vol_spike):
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below Donchian midpoint
            if close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: Price crosses above Donchian midpoint
            if close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
```

## Last Updated
2026-05-09 08:55
