# Strategy: 6h_VWAPDeviation_1dEMA34_Trend_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.156 | +15.2% | -10.2% | 59 | DISCARD |
| ETHUSDT | 0.209 | +30.6% | -10.0% | 74 | KEEP |
| SOLUSDT | -0.325 | -5.6% | -23.7% | 58 | DISCARD |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.151 | +7.6% | -5.5% | 25 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted Average Price (VWAP) Deviation + 1d Trend Filter + Volume Confirmation
# Price deviation from VWAP indicates mean reversion opportunities. 1d EMA34 ensures alignment with higher timeframe trend.
# Volume confirmation filters low-conviction moves. Designed for 12-37 trades/year on 6h to minimize fee drag.
# Works in bull markets via long when price < VWAP (discount) in uptrend and in bear markets via short when price > VWAP (premium) in downtrend.

name = "6h_VWAPDeviation_1dEMA34_Trend_VolumeConfirm"
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
    
    # Get 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h VWAP (Volume-Weighted Average Price)
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3
    # VWAP = cumulative(typical_price * volume) / cumulative(volume)
    cum_vol_price = np.cumsum(typical_price * volume)
    cum_vol = np.cumsum(volume)
    vwap = cum_vol_price / cum_vol
    
    # Calculate price deviation from VWAP as percentage
    vwap_deviation = (close - vwap) / vwap * 100
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(vwap_deviation[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price below VWAP (discount) AND 1d uptrend AND volume spike
            if (vwap_deviation[i] < -0.5 and  # Price at least 0.5% below VWAP
                close[i] > ema_34_aligned[i] and  # 1d uptrend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price above VWAP (premium) AND 1d downtrend AND volume spike
            elif (vwap_deviation[i] > 0.5 and   # Price at least 0.5% above VWAP
                  close[i] < ema_34_aligned[i] and  # 1d downtrend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses above VWAP OR 1d trend turns down
            if vwap_deviation[i] > 0 or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses below VWAP OR 1d trend turns up
            if vwap_deviation[i] < 0 or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-04 15:51
