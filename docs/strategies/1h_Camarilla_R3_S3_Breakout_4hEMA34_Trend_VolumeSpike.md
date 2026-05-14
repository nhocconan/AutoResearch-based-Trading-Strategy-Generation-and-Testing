# Strategy: 1h_Camarilla_R3_S3_Breakout_4hEMA34_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.155 | +27.3% | -8.8% | 200 | PASS |
| ETHUSDT | 0.001 | +18.6% | -12.5% | 192 | PASS |
| SOLUSDT | 0.308 | +43.4% | -23.5% | 171 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.326 | -4.8% | -9.6% | 73 | FAIL |
| ETHUSDT | 0.006 | +5.6% | -9.1% | 64 | PASS |
| SOLUSDT | -0.532 | -2.7% | -13.6% | 61 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R3 level AND price > EMA34(4h) AND volume > 2.0x 20-period average
# Short when price breaks below Camarilla S3 level AND price < EMA34(4h) AND volume > 2.0x 20-period average
# Exit when price crosses back below Camarilla S3 (for longs) or above R3 (for shorts) OR trend flips (price crosses EMA34(4h))
# Camarilla pivot levels provide intraday support/resistance structure proven effective on BTC/ETH
# 4h EMA34 provides higher timeframe trend filter to avoid counter-trend whipsaws in bear markets
# Volume spike confirms institutional participation
# Target: 15-37 trades/year per symbol (60-150 total over 4 years) for 1h timeframe
# Discrete sizing (0.20) to limit fee drag

name = "1h_Camarilla_R3_S3_Breakout_4hEMA34_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate EMA34 on 4h close for trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 4h EMA34 to 1h timeframe
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla levels: based on previous day's OHLC
    # R3 = close + 1.25*(high-low), S3 = close - 1.25*(high-low)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot levels
    camarilla_r3 = prev_close + 1.25 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.25 * (prev_high - prev_low)
    
    # Align daily Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND price > EMA34(4h) AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_4h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND price < EMA34(4h) AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_4h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses back below Camarilla S3 (mean reversion) OR price < EMA34(4h) (trend flip)
            if (close[i] < camarilla_s3_aligned[i] or 
                close[i] < ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses back above Camarilla R3 (mean reversion) OR price > EMA34(4h) (trend flip)
            if (close[i] > camarilla_r3_aligned[i] or 
                close[i] > ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
```

## Last Updated
2026-05-05 02:41
