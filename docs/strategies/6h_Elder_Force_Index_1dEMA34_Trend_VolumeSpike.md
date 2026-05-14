# Strategy: 6h_Elder_Force_Index_1dEMA34_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.312 | +34.3% | -11.6% | 36 | PASS |
| ETHUSDT | 0.418 | +42.8% | -11.3% | 34 | PASS |
| SOLUSDT | 1.215 | +189.8% | -15.7% | 37 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.837 | +0.6% | -5.4% | 11 | FAIL |
| ETHUSDT | 0.329 | +9.7% | -7.4% | 10 | PASS |
| SOLUSDT | -0.249 | +2.4% | -7.1% | 10 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h Elder Force Index (EFI) + 1d EMA34 Trend + Volume Spike Confirmation
Hypothesis: Elder Force Index combines price and volume to measure bull/bear power.
In trending markets (price > 1d EMA34), we take trades in direction of EFI(13) with volume confirmation.
In ranging markets (price near EMA34), we stay flat to avoid whipsaw.
Uses 6h primary timeframe with 1d EMA34 for higher timeframe trend filter.
Designed for BTC/ETH with 50-150 total trades over 4 years to minimize fee drag while capturing strong trends.
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
    
    # Get 1d data for EMA34 and EFI (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need 34 for EMA34
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Elder Force Index (EFI) on 1d: EFI = EMA(13) of (close * volume)
    # Typical price is not used; we use close * volume as force proxy
    force_1d = close_1d * pd.Series(df_1d['volume'])
    efi_13_1d = pd.Series(force_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    efi_13_1d_aligned = align_htf_to_ltf(prices, df_1d, efi_13_1d)
    
    # Calculate 20-period volume MA for volume spike confirmation (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34, EFI, and volume MA
    start_idx = max(34, 20)  # 34 for EMA34/EFI, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(efi_13_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_34_val = ema_34_1d_aligned[i]
        efi_val = efi_13_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        ranging = abs(curr_close - ema_34_val) / ema_34_val < 0.01  # within 1% of EMA34
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            if ranging:
                # Market ranging: stay flat
                signals[i] = 0.0
                position = 0
            elif uptrend:
                # Uptrend: look for long when EFI positive with volume
                long_signal = (efi_val > 0) and volume_confirm
                if long_signal:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
                    position = 0
            elif downtrend:
                # Downtrend: look for short when EFI negative with volume
                short_signal = (efi_val < 0) and volume_confirm
                if short_signal:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
                    position = 0
        elif position == 1:
            # Exit long: EFI turns negative OR price closes below EMA34
            if efi_val <= 0 or curr_close <= ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: EFI turns positive OR price closes above EMA34
            if efi_val >= 0 or curr_close >= ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Elder_Force_Index_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 03:37
