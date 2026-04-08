# Strategy: 6h_donchian20_12h_ema_vol_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.342 | +10.3% | -15.8% | 126 | DISCARD |
| ETHUSDT | 0.393 | +38.6% | -9.8% | 110 | KEEP |
| SOLUSDT | 0.113 | +25.0% | -21.4% | 93 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.634 | +13.6% | -7.6% | 40 | KEEP |
| SOLUSDT | -0.047 | +5.2% | -5.4% | 34 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA(50) filter and volume confirmation
# Enter long on breakout above 20-period high when price > 12h EMA(50) and volume > 2x average
# Enter short on breakdown below 20-period low when price < 12h EMA(50) and volume > 2x average
# Exit when price crosses 12-period EMA on 6h or opposite breakout occurs
# Targets 50-150 trades over 4 years with 0.25 position size to manage drawdown in both bull/bear markets

name = "6h_donchian20_12h_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.shift(1).values  # Use previous bar's high for breakout
    donchian_low = low_roll.shift(1).values    # Use previous bar's low for breakdown
    
    # 6-period EMA for exit signal
    ema_6 = pd.Series(close).ewm(span=6, adjust=False).mean().values
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below 6-period EMA OR breakdown signal
            if close[i] < ema_6[i] or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above 6-period EMA OR breakout signal
            if close[i] > ema_6[i] or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout/breakdown + trend filter + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > donchian_high[i] and close[i] > ema_50_12h_aligned[i]:
                    # Breakout above Donchian high with 12h uptrend
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_low[i] and close[i] < ema_50_12h_aligned[i]:
                    # Breakdown below Donchian low with 12h downtrend
                    signals[i] = -0.25
                    position = -1
    
    return signals
```

## Last Updated
2026-04-07 04:13
