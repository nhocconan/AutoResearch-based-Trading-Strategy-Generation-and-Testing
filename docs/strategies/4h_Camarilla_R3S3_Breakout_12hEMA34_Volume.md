# Strategy: 4h_Camarilla_R3S3_Breakout_12hEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.383 | +36.7% | -11.4% | 258 | PASS |
| ETHUSDT | 0.169 | +28.1% | -11.7% | 251 | PASS |
| SOLUSDT | 0.557 | +66.6% | -24.9% | 203 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.404 | -5.8% | -6.6% | 110 | FAIL |
| ETHUSDT | 1.190 | +23.1% | -7.1% | 96 | PASS |
| SOLUSDT | 0.681 | +15.2% | -8.0% | 66 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels from prior day identify key support/resistance; breakouts above R3 or below S3
# with volume confirmation indicate strong momentum. 12h EMA34 ensures trades align with intermediate trend
# to avoid false breakouts in choppy markets. Designed for 75-200 total trades over 4 years (19-50/year)
# on 4h timeframe. Works in bull markets (buying breakouts in uptrend) and bear markets
# (selling breakdowns in downtrend) by only taking trades in direction of 12h EMA34.

name = "4h_Camarilla_R3S3_Breakout_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate prior day's Camarilla levels (using 1d data)
    # Camarilla: based on prior day's high, low, close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prior_high = df_1d['high'].shift(1).values  # prior day's high
    prior_low = df_1d['low'].shift(1).values    # prior day's low
    prior_close = df_1d['close'].shift(1).values # prior day's close
    
    # Calculate Camarilla levels (R3/S3 are significant breakout levels)
    R3 = prior_close + (prior_high - prior_low) * 1.1 / 4
    S3 = prior_close - (prior_high - prior_low) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (wait for prior day to complete)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: 2.0x 20-period average (20*4h = ~3.3 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA34 and Camarilla)
    start_idx = max(34, 30)  # 34 bars for EMA34, 30 bars to ensure prior day data available
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R3 with volume spike AND price > 12h EMA34 (bullish trend)
            if (close[i] > R3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 with volume spike AND price < 12h EMA34 (bearish trend)
            elif (close[i] < S3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below R3 (failed breakout) OR price below 12h EMA34 (trend change)
            if close[i] < R3_aligned[i] or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above S3 (failed breakdown) OR price above 12h EMA34 (trend change)
            if close[i] > S3_aligned[i] or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 09:10
