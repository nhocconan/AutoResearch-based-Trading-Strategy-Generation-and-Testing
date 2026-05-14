# Strategy: 4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.190 | +30.8% | -16.2% | 64 | PASS |
| ETHUSDT | 0.064 | +19.5% | -22.0% | 66 | PASS |
| SOLUSDT | 0.883 | +189.9% | -30.2% | 52 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.947 | -6.5% | -12.6% | 23 | FAIL |
| ETHUSDT | 0.120 | +6.9% | -10.5% | 24 | PASS |
| SOLUSDT | -0.619 | -10.8% | -24.7% | 22 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Donchian breakouts capture institutional momentum with clear structure
# 12h EMA50 provides robust higher-timeframe trend alignment to avoid counter-trend trades
# Volume spike (>2.0 x 30-period EMA) confirms breakout validity while reducing false signals
# Discrete position sizing (0.30) balances opportunity with fee drag control
# Target: 100-180 total trades over 4 years (25-45/year) for optimal risk-adjusted returns

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation (volume spike > 2.0 x 30-period EMA)
    vol_ema_30 = pd.Series(volume).ewm(span=30, adjust=False, min_periods=30).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_30)
    
    # 12h data for Donchian channel and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Donchian(20) from previous 12h bar
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    lookback = 20
    prev_high_12h = pd.Series(df_12h['high']).shift(1).rolling(window=lookback, min_periods=lookback).max().values
    prev_low_12h = pd.Series(df_12h['low']).shift(1).rolling(window=lookback, min_periods=lookback).min().values
    
    # Align Donchian levels to 4h timeframe (wait for completed 12h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, prev_high_12h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, prev_low_12h)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Close breaks above Donchian upper with volume confirmation and uptrend
            if close[i] > donchian_upper_aligned[i] and volume_confirmation[i] and uptrend:
                signals[i] = 0.30
                position = 1
            # Short: Close breaks below Donchian lower with volume confirmation and downtrend
            elif close[i] < donchian_lower_aligned[i] and volume_confirmation[i] and downtrend:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close drops below Donchian lower (reversal) OR trend changes to downtrend
            if close[i] < donchian_lower_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: Close rises above Donchian upper (reversal) OR trend changes to uptrend
            if close[i] > donchian_upper_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
```

## Last Updated
2026-05-02 04:26
