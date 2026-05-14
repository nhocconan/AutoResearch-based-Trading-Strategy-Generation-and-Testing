# Strategy: 4h_Donchian20_1dEMA50_VolumeSpike_ATRStop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.143 | +26.3% | -11.7% | 134 | PASS |
| ETHUSDT | 0.218 | +31.0% | -15.1% | 126 | PASS |
| SOLUSDT | 0.634 | +79.3% | -20.8% | 119 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.745 | +0.1% | -4.3% | 47 | FAIL |
| ETHUSDT | 0.630 | +15.2% | -6.8% | 47 | PASS |
| SOLUSDT | 0.649 | +15.6% | -8.0% | 38 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend filter + volume spike confirmation + ATR trailing stop
- Uses 1d EMA50 as HTF trend filter for stronger regime identification
- Donchian(20) breakout captures momentum with proven edge on SOLUSDT
- Volume spike (2.0x 20-period MA) confirms institutional participation
- ATR(14) trailing stop (3.0x ATR) manages risk
- Discrete position sizing (0.25) minimizes fee churn
- Target: 20-40 trades/year per symbol (~80-160 total over 4 years)
- Works in bull markets (buying upper band breakouts in uptrend) and bear markets (selling lower band breakdowns in downtrend)
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
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Get 4h data for primary timeframe (Donchian, volume, ATR)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian channels (20-period) on 4h
    def donchian_channels(high_arr, low_arr, window=20):
        upper = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper_4h, donchian_lower_4h = donchian_channels(high_4h, low_4h, 20)
    
    # Calculate ATR(14) on 4h for volatility and trailing stop
    def atr(high_arr, low_arr, close_arr, window=14):
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period TR is just high-low
        atr_vals = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        return atr_vals
    
    atr_14_4h = atr(high_4h, low_4h, close_4h, 14)
    
    # Volume average (20-period) on 4h
    volume_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    highest_high_since_entry = 0.0  # For long trailing stop
    lowest_low_since_entry = 0.0    # For short trailing stop
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        ema_trend = ema50_1d_aligned[i]
        atr_val = atr_14_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend alignment
            # Long: price breaks above upper Donchian + volume spike + price > 1d EMA50 (uptrend)
            if price > upper and vol > 2.0 * vol_ma and price > ema_trend:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = price
            # Short: price breaks below lower Donchian + volume spike + price < 1d EMA50 (downtrend)
            elif price < lower and vol > 2.0 * vol_ma and price < ema_trend:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = price
        
        elif position == 1:
            # Update highest high for trailing stop
            if price > highest_high_since_entry:
                highest_high_since_entry = price
            
            # Exit long: price retracement to midpoint of Donchian channel OR ATR trailing stop
            mid_point = (upper + lower) / 2.0
            trailing_stop = highest_high_since_entry - 3.0 * atr_val
            
            if price < mid_point or price < trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest low for trailing stop
            if price < lowest_low_since_entry:
                lowest_low_since_entry = price
            
            # Exit short: price retracement to midpoint of Donchian channel OR ATR trailing stop
            mid_point = (upper + lower) / 2.0
            trailing_stop = lowest_low_since_entry + 3.0 * atr_val
            
            if price > mid_point or price > trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-17 21:15
