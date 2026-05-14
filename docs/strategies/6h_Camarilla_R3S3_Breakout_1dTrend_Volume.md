# Strategy: 6h_Camarilla_R3S3_Breakout_1dTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.429 | +36.7% | -5.9% | 177 | KEEP |
| ETHUSDT | 0.200 | +29.1% | -12.2% | 158 | KEEP |
| SOLUSDT | 0.692 | +80.6% | -18.3% | 139 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.778 | -0.2% | -4.3% | 75 | DISCARD |
| ETHUSDT | 1.815 | +31.3% | -6.2% | 58 | KEEP |
| SOLUSDT | 0.294 | +9.3% | -5.5% | 52 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla R3/S3 breakout with daily trend filter and volume spike
# Long when price breaks above R3, daily EMA(34) uptrend, and volume spike
# Short when price breaks below S3, daily EMA(34) downtrend, and volume spike
# Camarilla levels from prior day provide structured support/resistance
# Daily EMA filters for higher timeframe trend alignment
# Volume spike confirms institutional participation; avoids false breakouts
# Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    daily_close = df_1d['close'].values
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: R4 = C + (H-L)*1.5/2, R3 = C + (H-L)*1.25/2, S3 = C - (H-L)*1.25/2, S4 = C - (H-L)*1.5/2
    # where C = (H+L+C)/3 (typical price), but standard uses close of prior day
    # Using standard Camarilla formula based on prior day's range
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid division by zero and handle first bar
    price_range = prev_high - prev_low
    # Camarilla multipliers
    r3 = prev_close + price_range * 1.1 / 6  # R3 = C + (H-L)*1.1/6
    s3 = prev_close - price_range * 1.1 / 6  # S3 = C - (H-L)*1.1/6
    
    # Align Camarilla levels to 6h timeframe (available after daily close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1d_val = ema34_1d_aligned[i]
        price = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above R3, daily uptrend, volume spike
            if price > r3_val and price > ema34_1d_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3, daily downtrend, volume spike
            elif price < s3_val and price < ema34_1d_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below R3 or daily trend turns down
            if price < r3_val or price < ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above S3 or daily trend turns up
            if price > s3_val or price > ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 01:50
