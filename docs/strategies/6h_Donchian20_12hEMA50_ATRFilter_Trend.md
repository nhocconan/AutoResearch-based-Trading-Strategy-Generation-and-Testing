# Strategy: 6h_Donchian20_12hEMA50_ATRFilter_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.009 | +17.5% | -15.4% | 82 | FAIL |
| ETHUSDT | 0.191 | +31.2% | -15.5% | 80 | PASS |
| SOLUSDT | 0.873 | +164.6% | -29.9% | 86 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.074 | +6.1% | -13.5% | 27 | PASS |
| SOLUSDT | -0.169 | +0.5% | -16.5% | 26 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h trend filter and ATR-based volatility filter
# Uses 6h timeframe to capture medium-term moves with reasonable trade frequency
# Donchian breakouts capture momentum; 12h EMA50 ensures trend alignment
# ATR filter avoids low-volatility false breakouts
# Works in both bull and bear markets via trend-following logic
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Discrete position sizing: 0.25 (25% of capital) balances exposure and risk

name = "6h_Donchian20_12hEMA50_ATRFilter_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 12h Donchian channels (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian channels: upper = rolling max(high, 20), lower = rolling min(low, 20)
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to 6h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND price > 12h EMA50 (bullish trend) AND ATR > 0 (avoid zero volatility)
            if (close[i] > donch_high_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                atr[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low AND price < 12h EMA50 (bearish trend) AND ATR > 0 (avoid zero volatility)
            elif (close[i] < donch_low_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  atr[i] > 0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below Donchian low OR below 12h EMA50 (trend change)
            if close[i] < donch_low_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high OR above 12h EMA50 (trend change)
            if close[i] > donch_high_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 10:17
