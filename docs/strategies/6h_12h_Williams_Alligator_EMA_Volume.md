# Strategy: 6h_12h_Williams_Alligator_EMA_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.092 | +12.1% | -16.2% | 96 | FAIL |
| ETHUSDT | 0.011 | +16.4% | -22.0% | 96 | PASS |
| SOLUSDT | 0.786 | +144.6% | -30.3% | 95 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.420 | +13.8% | -7.9% | 37 | PASS |
| SOLUSDT | -0.206 | -0.7% | -12.7% | 34 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 12h EMA trend filter + volume confirmation.
# Williams Alligator uses smoothed SMAs (Jaws=13, Teeth=8, Lips=5).
# Long: Lips > Teeth > Jaws (bullish alignment) + price > 12h EMA50 + volume > 1.5x avg volume.
# Short: Lips < Teeth < Jaws (bearish alignment) + price < 12h EMA50 + volume > 1.5x avg volume.
# Works in both bull and bear by using 12h EMA50 as trend filter and requiring volume confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for Williams Alligator and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Williams Alligator: smoothed SMAs
    # Jaws: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        sma = np.nansum(arr[:period]) / period
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaws = smma((high_12h + low_12h) / 2, 13)
    teeth = smma((high_12h + low_12h) / 2, 8)
    lips = smma((high_12h + low_12h) / 2, 5)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Average volume (10-period = 5 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(10, n):
        avg_volume[i] = np.mean(volume[i-10:i])
    
    # Align 12h indicators to 6h timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_12h, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(10, n):
        # Skip if any required data is not ready
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        jaw = jaws_aligned[i]
        tooth = teeth_aligned[i]
        lip = lips_aligned[i]
        ema_trend = ema_50_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        # Williams Alligator signals
        bullish_alignment = lip > tooth > jaw
        bearish_alignment = lip < tooth < jaw
        
        if position == 0:
            # Long: bullish alignment + above EMA50 + volume confirmation
            if (bullish_alignment and 
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: bearish alignment + below EMA50 + volume confirmation
            elif (bearish_alignment and 
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bearish alignment or price below EMA50
            if (bearish_alignment or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: bullish alignment or price above EMA50
            if (bullish_alignment or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12h_Williams_Alligator_EMA_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-13 22:38
