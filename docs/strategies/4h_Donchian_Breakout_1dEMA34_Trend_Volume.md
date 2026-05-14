# Strategy: 4h_Donchian_Breakout_1dEMA34_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.112 | +25.2% | -19.4% | 110 | PASS |
| ETHUSDT | 0.124 | +25.8% | -17.8% | 112 | PASS |
| SOLUSDT | 0.713 | +113.4% | -25.3% | 106 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.268 | -7.5% | -10.2% | 49 | FAIL |
| ETHUSDT | 0.104 | +6.9% | -10.6% | 38 | PASS |
| SOLUSDT | 0.493 | +15.1% | -12.4% | 33 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation
# Uses 1d EMA34 for trend direction (long when price > EMA34, short when price < EMA34)
# and Donchian(20) channel breakouts for entries. Volume > 1.8x 24-period average confirms strength.
# Trend filter avoids counter-trend trades, Donchian provides clear breakout signals.
# Target: 20-30 trades/year to minimize fee decay while capturing strong momentum.
# Focus on BTC/ETH as primary assets with proven Donchian edge from DB.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) on 4h data
    donchian_period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(donchian_period, n):
        upper[i] = np.max(high[i-donchian_period:i])
        lower[i] = np.min(low[i-donchian_period:i])
    
    # 24-period average volume for spike detection (4h bars in 1d = 6, so 24 = 4d)
    vol_ma = np.full(n, np.nan)
    vol_period = 24
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(donchian_period, vol_period, 1)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(upper[i]) or
            np.isnan(lower[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine trend from 1d EMA34
        uptrend = price > ema_34_1d_aligned[i]
        downtrend = price < ema_34_1d_aligned[i]
        
        # Volume confirmation: spike > 1.8x average
        volume_confirmation = vol_ratio > 1.8
        
        if position == 0:
            # Long breakout: price breaks above upper Donchian band in uptrend
            if uptrend and price > upper[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short breakdown: price breaks below lower Donchian band in downtrend
            elif downtrend and price < lower[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below lower Donchian band or trend reverses
            if price < lower[i] or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above upper Donchian band or trend reverses
            if price > upper[i] or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-27 14:13
