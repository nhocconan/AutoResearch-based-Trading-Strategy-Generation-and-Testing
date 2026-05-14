# Strategy: 12h_Camarilla_R3S3_1dEMA34_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.042 | +17.1% | -14.1% | 48 | DISCARD |
| ETHUSDT | 0.033 | +20.4% | -12.7% | 46 | KEEP |
| SOLUSDT | 0.153 | +27.7% | -20.8% | 35 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.601 | +16.8% | -10.6% | 13 | KEEP |
| SOLUSDT | -0.215 | +0.2% | -16.6% | 12 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter (EMA34) and volume confirmation (>1.5x 20 EMA volume)
# Uses 12h Camarilla pivot levels (R3/S3) for structure - captures strong momentum bursts after consolidation
# 1d EMA34 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation filters false breakouts (>1.5x average volume) - balanced to avoid overtrading
# Discrete sizing 0.25 minimizes fee churn while maintaining profitability
# Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe
# Works in bull markets (continuation at R3) and bear markets (continuation at S3)
# Focus on BTC/ETH by requiring 1d trend alignment (avoids SOL-only bias)

name = "12h_Camarilla_R3S3_1dEMA34_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA34 calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34) trend filter from prior completed 1d bar
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_shifted = np.roll(ema_34_1d, 1)
    ema_34_1d_shifted[0] = np.nan
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 12h Camarilla levels (R3, S3) from prior completed 12h bar
    # Typical Price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    tp_series = pd.Series(typical_price)
    
    # Calculate pivot point (PP) from prior 12h bar
    pp = tp_series.rolling(window=1, min_periods=1).mean().values  # PP = typical price of prior bar
    pp_shifted = np.roll(pp, 1)
    pp_shifted[0] = np.nan
    
    # Calculate R3 and S3 levels: R3 = PP + 2*(High - Low), S3 = PP - 2*(High - Low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    hl_range = high_series - low_series
    
    r3 = pp_shifted + 2.0 * hl_range
    s3 = pp_shifted - 2.0 * hl_range
    
    # Shift by 1 to use only prior completed 12h bar (no look-ahead)
    r3_shifted = np.roll(r3, 1)
    s3_shifted = np.roll(s3, 1)
    r3_shifted[0] = np.nan
    s3_shifted[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(r3_shifted[i]) or np.isnan(s3_shifted[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND price > 1d EMA34 AND volume spike
            if close[i] > r3_shifted[i] and close[i] > ema_34_1d_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 AND price < 1d EMA34 AND volume spike
            elif close[i] < s3_shifted[i] and close[i] < ema_34_1d_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to S3 OR price crosses below 1d EMA34
            if close[i] < s3_shifted[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to R3 OR price crosses above 1d EMA34
            if close[i] > r3_shifted[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-05 00:02
