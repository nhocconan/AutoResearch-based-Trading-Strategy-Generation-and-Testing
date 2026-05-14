# Strategy: 6h_Camarilla_R3S3_Breakout_12hEMA50_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.090 | +23.8% | -14.4% | 82 | PASS |
| ETHUSDT | 0.343 | +44.6% | -18.2% | 79 | PASS |
| SOLUSDT | 0.904 | +169.2% | -28.8% | 73 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.012 | -7.1% | -12.2% | 37 | FAIL |
| ETHUSDT | 0.452 | +14.6% | -9.4% | 28 | PASS |
| SOLUSDT | -0.393 | -4.4% | -23.0% | 26 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 level AND 12h close > 12h EMA50 AND volume > 2.0x 20-period average
# Short when price breaks below Camarilla S3 level AND 12h close < 12h EMA50 AND volume > 2.0x 20-period average
# Exit when price crosses 12h EMA50 (trend reversal)
# Uses 6h primary timeframe with 12h HTF for trend filter
# Camarilla levels provide intraday support/resistance that work in ranging and trending markets
# Volume confirmation reduces false breakouts, trend filter ensures alignment with higher timeframe direction
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels on 6h data (based on previous bar's range)
    if len(high) >= 2 and len(low) >= 2:
        # Camarilla levels calculated from previous bar's high-low range
        prev_high = np.roll(high, 1)
        prev_low = np.roll(low, 1)
        prev_close = np.roll(close, 1)
        prev_high[0] = np.nan
        prev_low[0] = np.nan
        prev_close[0] = np.nan
        
        range_val = prev_high - prev_low
        camarilla_r3 = prev_close + (range_val * 1.1 / 4)
        camarilla_s3 = prev_close - (range_val * 1.1 / 4)
    else:
        camarilla_r3 = np.full(n, np.nan)
        camarilla_s3 = np.full(n, np.nan)
    
    # Volume confirmation: volume > 2.0x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND 12h close > 12h EMA50 AND volume spike
            if (close[i] > camarilla_r3[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND 12h close < 12h EMA50 AND volume spike
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 12h EMA50 (trend reversal)
            if close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 12h EMA50 (trend reversal)
            if close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-05 06:25
