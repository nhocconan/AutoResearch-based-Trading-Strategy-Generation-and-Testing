# Strategy: 4h_Donchian20_Breakout_12hTrend_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.017 | +19.9% | -17.9% | 106 | PASS |
| ETHUSDT | 0.429 | +50.3% | -12.1% | 96 | PASS |
| SOLUSDT | 0.984 | +177.0% | -30.0% | 98 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.374 | -8.2% | -9.7% | 43 | FAIL |
| ETHUSDT | 0.162 | +8.0% | -9.8% | 39 | PASS |
| SOLUSDT | 0.100 | +6.7% | -16.0% | 34 | PASS |

## Code
```python
#!/usr/bin/env python3
name = "4h_Donchian20_Breakout_12hTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA50 trend
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    trend_up = close > ema_50_12h_aligned
    trend_down = close < ema_50_12h_aligned
    
    # Donchian(20) channels on 4h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume spike: current volume > 2.0x 20-period average (~3.3 days)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # ~1.5 days (3*4h) to prevent overtrading
    
    start_idx = max(20, 50)  # Ensure enough data for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction
        trending_up = trend_up[i]
        trending_down = trend_down[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above Donchian high with volume spike in 12h uptrend
            if (close[i] > donchian_high[i] and 
                trending_up and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below Donchian low with volume spike in 12h downtrend
            elif (close[i] < donchian_low[i] and 
                  trending_down and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls back below Donchian low or 12h trend changes to down
            if close[i] < donchian_low[i] or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises back above Donchian high or 12h trend changes to up
            if close[i] > donchian_high[i] or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: On 4h timeframe, price breaking above/below Donchian(20) channels with volume spike confirmation and 12h EMA50 trend filter captures institutional breakout momentum. Donchian channels represent dynamic support/resistance, reducing false breakouts. 12h trend filter ensures alignment with higher timeframe momentum. Volume spike filter (2.0x 20-period average) confirms institutional participation. Cooldown period prevents overtrading. Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag. Works in bull markets (breakouts above Donchian high in 12h uptrend) and bear markets (breakdowns below Donchian low in 12h downtrend). Uses discrete position sizing (0.25) to balance risk and reward while reducing fee churn. This strategy avoids saturated Camarilla patterns and focuses on proven Donchian breakout with volume/trend confluence, which has shown strong performance in DB (e.g., 4h_Donchian20_Breakout_12hTrend_Volume with 0.573 avg Sharpe).
```

## Last Updated
2026-05-07 13:03
