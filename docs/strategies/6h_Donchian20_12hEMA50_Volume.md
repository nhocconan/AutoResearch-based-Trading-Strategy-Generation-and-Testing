# Strategy: 6h_Donchian20_12hEMA50_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.278 | +4.8% | -18.3% | 78 | DISCARD |
| ETHUSDT | 0.119 | +25.4% | -12.2% | 74 | KEEP |
| SOLUSDT | 0.899 | +148.6% | -20.5% | 62 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.423 | +13.3% | -9.0% | 25 | KEEP |
| SOLUSDT | -0.282 | -1.1% | -19.2% | 22 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Long: Close breaks above upper band AND price > 12h EMA50 (uptrend) AND volume > 2.0x 20-period MA
# Short: Close breaks below lower band AND price < 12h EMA50 (downtrend) AND volume > 2.0x 20-period MA
# Exit: Opposite Donchian breakout or EMA50 trend reversal.
# Discrete sizing 0.25. Target: 80-180 total trades over 4 years (20-45/year).
# Donchian channels provide clear structure; 12h EMA50 filters higher timeframe trend;
# volume confirmation reduces false signals. Works in bull via long signals with trend alignment
# and in bear via short signals with trend alignment.

name = "6h_Donchian20_12hEMA50_Volume"
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian(20) channels using previous 20 periods (excluding current)
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume regime: current 6h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(high_ma_20[i]) or np.isnan(low_ma_20[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            # Long: Close breaks above upper band AND uptrend AND volume spike
            if close_val > high_ma_20[i] and is_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below lower band AND downtrend AND volume spike
            elif close_val < low_ma_20[i] and is_downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close breaks below lower band OR trend turns down
            if close_val < low_ma_20[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close breaks above upper band OR trend turns up
            if close_val > high_ma_20[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-08 16:46
