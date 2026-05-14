# Strategy: 4h_1d_camarilla_breakout_v32

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.024 | +22.1% | -4.5% | 88 | PASS |
| ETHUSDT | 0.003 | +21.1% | -5.9% | 63 | PASS |
| SOLUSDT | -0.308 | +5.6% | -22.5% | 61 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.916 | -4.1% | -6.6% | 31 | FAIL |
| ETHUSDT | 0.726 | +12.1% | -7.0% | 27 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and ADX trend filter
# Uses daily pivot levels (S4/R4) from prior day's OHLC for mean-reversion entries in ranging markets
# ADX(14) > 25 filters for trending conditions to avoid false breakouts
# Volume > 2x 4-period average confirms institutional participation
# Fixed position size 0.25 to limit drawdown and control risk
# Designed for 4-6 trades per month (~50-75/year) to minimize fee drag

name = "4h_1d_camarilla_breakout_v32"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (using prior day's OHLC)
    r4 = np.full(len(df_1d), np.nan)
    s4 = np.full(len(df_1d), np.nan)
    prev_high = np.full(len(df_1d), np.nan)
    prev_low = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        ph = float(df_1d['high'].iloc[i-1])
        pl = float(df_1d['low'].iloc[i-1])
        pc = float(df_1d['close'].iloc[i-1])
        r4[i] = pc + (ph - pl) * 1.1 / 2
        s4[i] = pc - (ph - pl) * 1.1 / 2
        prev_high[i] = ph
        prev_low[i] = pl
    
    # Align 1d values to 4h timeframe
    r4_4h = align_htf_to_ltf(prices, df_1d, r4)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4)
    prev_high_4h = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_4h = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Volume confirmation: 4-period average (16h)
    vol_ma_4 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 4:
            vol_sum -= volume[i-4]
        if i >= 3:
            vol_ma_4[i] = vol_sum / 4
    
    # ADX calculation (14-period)
    dx = np.full(n, np.nan)
    tr = np.full(n, np.nan)
    dm_plus = np.full(n, np.nan)
    dm_minus = np.full(n, np.nan)
    
    for i in range(1, n):
        # True Range
        tr0 = high[i] - low[i]
        tr1 = abs(high[i] - close[i-1])
        tr2 = abs(low[i] - close[i-1])
        tr[i] = max(tr0, tr1, tr2)
        
        # Directional Movement
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        else:
            dm_plus[i] = 0.0
        if down_move > up_move and down_move > 0:
            dm_minus[i] = down_move
        else:
            dm_minus[i] = 0.0
    
    # Smoothed averages
    tr14 = np.full(n, np.nan)
    dm_plus_14 = np.full(n, np.nan)
    dm_minus_14 = np.full(n, np.nan)
    
    # Initial values
    if n >= 14:
        tr14[13] = np.nansum(tr[1:14])
        dm_plus_14[13] = np.nansum(dm_plus[1:14])
        dm_minus_14[13] = np.nansum(dm_minus[1:14])
        
        # Wilder smoothing
        for i in range(14, n):
            tr14[i] = tr14[i-1] - (tr14[i-1] / 14) + tr[i]
            dm_plus_14[i] = dm_plus_14[i-1] - (dm_plus_14[i-1] / 14) + dm_plus[i]
            dm_minus_14[i] = dm_minus_14[i-1] - (dm_minus_14[i-1] / 14) + dm_minus[i]
    
    # DI and DX
    di_plus = np.full(n, np.nan)
    di_minus = np.full(n, np.nan)
    for i in range(14, n):
        if tr14[i] > 0:
            di_plus[i] = 100 * dm_plus_14[i] / tr14[i]
            di_minus[i] = 100 * dm_minus_14[i] / tr14[i]
            dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX (smoothed DX)
    adx = np.full(n, np.nan)
    if n >= 28:
        adx[27] = np.nansum(dx[14:28]) / 14
        for i in range(28, n):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(28, n):  # Start after ADX warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_4h[i]) or 
            np.isnan(s4_4h[i]) or 
            np.isnan(prev_high_4h[i]) or 
            np.isnan(prev_low_4h[i]) or 
            np.isnan(vol_ma_4[i]) or 
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes back inside previous day's range OR ADX drops below 20 (trend weakening)
            if (close[i] <= prev_high_4h[i] and close[i] >= prev_low_4h[i]) or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes back inside previous day's range OR ADX drops below 20
            if (close[i] <= prev_high_4h[i] and close[i] >= prev_low_4h[i]) or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above R4 with volume confirmation AND ADX > 25 (trending)
            vol_ratio = volume[i] / vol_ma_4[i] if vol_ma_4[i] > 0 else 0
            if (close[i] > r4_4h[i] and 
                vol_ratio > 2.0 and 
                adx[i] > 25):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below S4 with volume confirmation AND ADX > 25
            elif (close[i] < s4_4h[i] and 
                  vol_ratio > 2.0 and 
                  adx[i] > 25):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 10:28
