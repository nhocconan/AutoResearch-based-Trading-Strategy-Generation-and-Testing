# Strategy: 4H_Donchian20_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.320 | +34.0% | -11.1% | 135 | PASS |
| ETHUSDT | 0.414 | +42.4% | -12.9% | 129 | PASS |
| SOLUSDT | 0.572 | +70.9% | -22.7% | 122 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.196 | -3.5% | -5.1% | 57 | FAIL |
| ETHUSDT | 0.830 | +18.3% | -5.1% | 45 | PASS |
| SOLUSDT | -0.088 | +4.2% | -9.5% | 39 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above 4h Donchian upper channel (20) AND price > 1d EMA34 (uptrend) AND volume > 2.0x average.
Short when price breaks below 4h Donchian lower channel (20) AND price < 1d EMA34 (downtrend) AND volume > 2.0x average.
Exit when price reverts to 4h Donchian middle (20) or trend reverses (price crosses 1d EMA34).
Uses 4h timeframe with tight entry conditions to avoid fee drag. Donchian channels provide clear breakout structure.
1d EMA34 provides stable trend filter. Volume confirmation ensures high-conviction breakouts.
Target: 75-150 trades over 4 years (19-37/year) to stay within proven working range.
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
    
    # Calculate 4h Donchian channels (20-period) - using primary timeframe data
    # We need to calculate on 4h data but we only have 15m/1h etc. - so we'll use rolling on current timeframe
    # However, for true 4h Donchian we need to use 4h data via mtf_data
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian channels on 4h
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2.0
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 4h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    middle_20_aligned = align_htf_to_ltf(prices, df_4h, middle_20)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (20-period) on 4h timeframe
    # We need volume data aligned to 4h - get 4h volume
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or np.isnan(middle_20_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_val = upper_20_aligned[i]
        lower_val = lower_20_aligned[i]
        middle_val = middle_20_aligned[i]
        ema34_val = ema34_1d_aligned[i]
        vol_ma_val = vol_ma_4h_aligned[i]
        # Get current 4h-aligned price and volume
        # We need to get the 4h-aligned close and volume for the current bar
        # Since we're iterating on the primary timeframe, we use the aligned arrays
        # For simplicity, we'll use the current close/volume from prices (which is aligned to primary TF)
        # But we need to ensure we're checking 4h breakout conditions
        # The aligned arrays already give us the 4h values at each primary timeframe bar
        
        # For volume, we need to check if current 4h bar volume > 2x average
        # But we don't have current 4h volume in the loop - we'll use price-based volume confirmation
        # Instead, we'll use the current primary timeframe volume with a longer average
        vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_current = volume[i]
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above 4h upper channel AND price > 1d EMA34 (uptrend) AND volume confirmation
            if (price > upper_val and price > ema34_val and vol_current > 2.0 * vol_ma_primary[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 4h lower channel AND price < 1d EMA34 (downtrend) AND volume confirmation
            elif (price < lower_val and price < ema34_val and vol_current > 2.0 * vol_ma_primary[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to middle channel OR price breaks below 1d EMA34 (trend reversal)
                if price <= middle_val or price < ema34_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to middle channel OR price breaks above 1d EMA34 (trend reversal)
                if price >= middle_val or price > ema34_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 01:40
