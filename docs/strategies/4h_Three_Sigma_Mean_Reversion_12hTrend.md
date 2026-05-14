# Strategy: 4h_Three_Sigma_Mean_Reversion_12hTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.252 | +12.3% | -7.9% | 59 | FAIL |
| ETHUSDT | 0.079 | +23.5% | -11.2% | 58 | PASS |
| SOLUSDT | -0.111 | +9.2% | -22.2% | 59 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.152 | +7.6% | -8.6% | 29 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_Three_Sigma_Mean_Reversion_12hTrend
# Hypothesis: Price reverts to mean after extreme deviations (>3σ) from 12h VWAP, 
# but only in the direction of the 12h trend (EMA50). Volume spike confirms conviction.
# Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
# Low frequency due to strict 3σ threshold and volume confirmation.

name = "4h_Three_Sigma_Mean_Reversion_12hTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 12h data for VWAP and trend
    df_12h = get_htf_data(prices, '12h')
    
    # Typical price for VWAP calculation
    typical_price = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    # VWAP: cumulative TP * volume / cumulative volume
    vwap = (typical_price * df_12h['volume']).cumsum() / df_12h['volume'].cumsum()
    vwap_vals = vwap.values
    
    # Standard deviation of typical price from VWAP (20-period rolling)
    price_dev = typical_price - vwap
    std_dev = price_dev.rolling(window=20, min_periods=20).std().values
    
    # 12h trend filter: EMA50
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 12h indicators to 4h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_12h, vwap_vals)
    std_dev_aligned = align_htf_to_ltf(prices, df_12h, std_dev)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike: volume > 2.0 * 30-period average (high threshold for low frequency)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > 2.0 * vol_ma_30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(vwap_aligned[i]) or 
            np.isnan(std_dev_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Deviation from VWAP in standard deviation units
        if std_dev_aligned[i] > 0:
            z_score = (close[i] - vwap_aligned[i]) / std_dev_aligned[i]
        else:
            z_score = 0
        
        # Trend conditions
        uptrend = close[i] > ema50_12h_aligned[i]
        downtrend = close[i] < ema50_12h_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]

        if position == 0:
            # LONG: Price < VWAP - 3σ (oversold) + uptrend + volume spike
            if z_score < -3.0 and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Price > VWAP + 3σ (overbought) + downtrend + volume spike
            elif z_score > 3.0 and downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above VWAP OR trend reversal
            if close[i] >= vwap_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below VWAP OR trend reversal
            if close[i] <= vwap_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-13 02:25
