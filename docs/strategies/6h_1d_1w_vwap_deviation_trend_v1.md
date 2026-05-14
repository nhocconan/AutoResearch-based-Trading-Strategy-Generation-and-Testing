# Strategy: 6h_1d_1w_vwap_deviation_trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.126 | +18.4% | -5.6% | 3 | FAIL |
| ETHUSDT | 0.141 | +26.3% | -12.4% | 11 | PASS |
| SOLUSDT | -0.402 | -11.9% | -50.4% | 6 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.513 | +11.8% | -8.4% | 3 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Institutional Flow Detector using 1d Volume Weighted Average Price (VWAP) deviation and 1w trend filter.
# Long when price deviates below 1d VWAP by >1.5σ and 1w trend is up; short when price deviates above 1d VWAP by >1.5σ and 1w trend is down.
# Uses volume-weighted price deviation to detect institutional accumulation/distribution.
# 1w trend filter prevents counter-trend trading. Designed for 15-30 trades/year on 6h timeframe.

name = "6h_1d_1w_vwap_deviation_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1d VWAP (typical price * volume) / volume
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_values = vwap_1d.values
    
    # Align 1d VWAP to 6h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_values)
    
    # Calculate 1d VWAP standard deviation (20-period)
    typical_price_1d_series = pd.Series(typical_price_1d.values)
    volume_1d_series = pd.Series(df_1d['volume'].values)
    vwap_deviation = typical_price_1d_series - vwap_1d
    vwap_std_20 = vwap_deviation.rolling(window=20, min_periods=20).std().values
    vwap_std_20_aligned = align_htf_to_ltf(prices, df_1d, vwap_std_20)
    
    # Calculate 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after VWAP std period
        # Skip if any required data is invalid
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(vwap_std_20_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Price deviation from 1d VWAP in standard deviations
        price_dev = (close[i] - vwap_1d_aligned[i]) / vwap_std_20_aligned[i] if vwap_std_20_aligned[i] > 0 else 0
        
        # Determine 1w trend direction
        is_uptrend = close[i] > ema_20_1w_aligned[i]
        is_downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Entry conditions: significant deviation from VWAP with trend alignment
        # Long when price is significantly below VWAP (accumulation) and 1w trend up
        # Short when price is significantly above VWAP (distribution) and 1w trend down
        vwap_long_signal = price_dev < -1.5 and is_uptrend
        vwap_short_signal = price_dev > 1.5 and is_downtrend
        
        # Exit conditions: price returns toward VWAP
        exit_long = price_dev > -0.5  # Return to within 0.5σ of VWAP
        exit_short = price_dev < 0.5   # Return to within 0.5σ of VWAP
        
        # Priority: entry > exit > hold
        if vwap_long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif vwap_short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
```

## Last Updated
2026-04-11 23:06
