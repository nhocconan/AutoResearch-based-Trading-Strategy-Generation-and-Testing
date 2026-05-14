# Strategy: 4h_VWAP_DailyTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.093 | +24.1% | -12.8% | 60 | PASS |
| ETHUSDT | 0.303 | +38.1% | -17.0% | 57 | PASS |
| SOLUSDT | 0.975 | +163.6% | -20.1% | 53 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.166 | -3.7% | -6.2% | 26 | FAIL |
| ETHUSDT | 0.358 | +11.6% | -10.8% | 26 | PASS |
| SOLUSDT | -0.319 | -2.1% | -17.3% | 21 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Volume Weighted Average Price (VWAP) with 1-day trend filter and volume confirmation
# Long when price > VWAP and daily EMA(34) uptrend and volume spike
# Short when price < VWAP and daily EMA(34) downtrend and volume spike
# VWAP acts as dynamic support/resistance; daily EMA provides higher timeframe bias
# Volume spike confirms institutional participation; avoids choppy false breakouts
# Targets 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

name = "4h_VWAP_DailyTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    daily_close = df_1d['close'].values
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate VWAP: cumulative (price * volume) / cumulative volume
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume)
    # Avoid division by zero
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1d_val = ema34_1d_aligned[i]
        price = close[i]
        vwap_val = vwap[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price > VWAP and daily uptrend and volume spike
            if price > vwap_val and price > ema34_1d_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price < VWAP and daily downtrend and volume spike
            elif price < vwap_val and price < ema34_1d_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < VWAP or daily trend turns down
            if price < vwap_val or price < ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > VWAP or daily trend turns up
            if price > vwap_val or price > ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-08 13:16
