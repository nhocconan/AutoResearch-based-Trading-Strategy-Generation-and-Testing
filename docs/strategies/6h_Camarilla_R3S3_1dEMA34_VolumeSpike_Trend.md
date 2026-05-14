# Strategy: 6h_Camarilla_R3S3_1dEMA34_VolumeSpike_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.256 | +35.1% | -15.5% | 66 | PASS |
| ETHUSDT | 0.223 | +33.9% | -17.0% | 69 | PASS |
| SOLUSDT | 0.888 | +169.4% | -32.0% | 64 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.385 | +0.3% | -8.5% | 29 | FAIL |
| ETHUSDT | 0.790 | +23.4% | -8.5% | 20 | PASS |
| SOLUSDT | 0.270 | +10.7% | -13.5% | 20 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d trend filter and volume spike confirmation.
# Uses 6h primary timeframe to reduce trade frequency vs lower TFs, targeting 12-37 trades/year.
# Camarilla levels from 1d provide institutional structure, 1d EMA34 filters trend direction,
# and volume spike confirms momentum. Designed for BTC/ETH to work in both bull and bear markets
# by taking breakouts in the direction of the higher timeframe trend.

name = "6h_Camarilla_R3S3_1dEMA34_VolumeSpike_Trend"
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
    
    # Get 1d data for Camarilla pivots and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    typical_price_values = typical_price.values
    
    # Camarilla levels: R3 = PP + (H - L) * 1.1/4, S3 = PP - (H - L) * 1.1/4
    # Using previous day's values (already completed bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Pivot point
    pp = (high_1d + low_1d + close_1d_vals) / 3
    # Range
    rng = high_1d - low_1d
    # Camarilla levels
    r3 = pp + (rng * 1.1 / 4)
    s3 = pp - (rng * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe (wait for 1d bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate volume regime: current 6h volume > 2.0x 20-period MA (strict to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_trend = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(r3_val) or np.isnan(s3_val) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Entry conditions
        # Long: break above R3 with volume spike and above 1d EMA34
        long_entry = (close[i] > r3_val) and vol_spike and (close[i] > ema_trend)
        # Short: break below S3 with volume spike and below 1d EMA34
        short_entry = (close[i] < s3_val) and vol_spike and (close[i] < ema_trend)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on close below EMA34 (trend change)
            if close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on close above EMA34 (trend change)
            if close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-03 04:45
