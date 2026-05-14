# Strategy: 6h_Donchian20_Breakout_12hEMA50_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.278 | +4.8% | -18.3% | 78 | FAIL |
| ETHUSDT | 0.119 | +25.4% | -12.2% | 74 | PASS |
| SOLUSDT | 0.899 | +148.6% | -20.5% | 62 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.423 | +13.3% | -9.0% | 25 | PASS |
| SOLUSDT | -0.282 | -1.1% | -19.2% | 22 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Uses 12h EMA50 for trend filter (intermediate trend) and 6h Donchian channels for breakout signals
# Entry logic: Long when price breaks above 6h Donchian upper channel with volume spike and price > 12h EMA50
#              Short when price breaks below 6h Donchian lower channel with volume spike and price < 12h EMA50
# Exit logic: Exit when price crosses the 12h EMA50 (trend reversal) or opposite Donchian channel
# Works in both bull and bear markets by trading with the 12h trend
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Discrete sizing 0.25 balances profit potential and fee drag

name = "6h_Donchian20_Breakout_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 6h Donchian channels (20-period)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Donchian channels: upper = max(high, 20), lower = min(low, 20)
    donchian_upper = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 6h timeframe (use previous completed 6h bar's channels)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Break above 6h Donchian upper AND price > 12h EMA50 (uptrend) AND volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Break below 6h Donchian lower AND price < 12h EMA50 (downtrend) AND volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below 12h EMA50 (trend change) OR break below 6h Donchian lower (reversal)
            if (close[i] < ema_50_12h_aligned[i] or 
                close[i] < donchian_lower_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above 12h EMA50 (trend change) OR break above 6h Donchian upper (reversal)
            if (close[i] > ema_50_12h_aligned[i] or 
                close[i] > donchian_upper_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 07:23
