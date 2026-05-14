# Strategy: 4h_Donchian20_12hEMA50_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.036 | +16.3% | -19.6% | 136 | FAIL |
| ETHUSDT | 0.050 | +20.0% | -15.8% | 126 | PASS |
| SOLUSDT | 0.951 | +178.6% | -25.8% | 132 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.135 | +7.4% | -9.3% | 44 | PASS |
| SOLUSDT | 0.405 | +13.4% | -13.5% | 44 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout (20-period) with 12h trend filter (EMA50) and volume confirmation.
# Uses Donchian channels for breakout detection, 12h EMA50 for trend direction, and volume spike for confirmation.
# Designed to capture strong trending moves while avoiding false breakouts in choppy markets.
# Target: 20-40 trades/year (80-160 total over 4 years).
name = "4h_Donchian20_12hEMA50_VolumeConfirm"
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 12h close
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate volume average (20-period) for spike detection
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need 50 periods for EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema = ema_50_12h_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol = volume[i]
        vol_avg = vol_ma[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high AND price > 12h EMA50 (uptrend) AND volume > 1.5x average
            if price > upper and price > ema and vol > 1.5 * vol_avg:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low AND price < 12h EMA50 (downtrend) AND volume > 1.5x average
            elif price < lower and price < ema and vol > 1.5 * vol_avg:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian low OR trend reverses (price < 12h EMA50)
            if price < lower or price < ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian high OR trend reverses (price > 12h EMA50)
            if price > upper or price > ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 02:37
