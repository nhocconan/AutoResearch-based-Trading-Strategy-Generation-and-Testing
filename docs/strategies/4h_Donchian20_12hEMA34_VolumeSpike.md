# Strategy: 4h_Donchian20_12hEMA34_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.000 | +17.4% | -22.7% | 117 | PASS |
| ETHUSDT | 0.521 | +67.0% | -14.4% | 109 | PASS |
| SOLUSDT | 0.907 | +193.3% | -31.0% | 107 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.420 | -12.2% | -14.1% | 55 | FAIL |
| ETHUSDT | 0.186 | +8.5% | -11.7% | 45 | PASS |
| SOLUSDT | 0.150 | +7.6% | -19.3% | 38 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA34 trend + volume spike
# Targets 75-200 total trades over 4 years (19-50/year) to minimize fee drag
# Donchian(20) provides clear breakout structure on 4h timeframe
# 12h EMA34 determines trend bias: long when price > EMA34, short when price < EMA34
# Volume spike (2x 20-period average) confirms institutional participation
# Works in bull markets via breakouts with trend alignment and bear markets via fade of false breakouts
# Discrete position sizing: 0.30 (30% of capital) balances exposure and risk
# Uses 12h as HTF as specified in experiment #117273

name = "4h_Donchian20_12hEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian(20) channels (prior completed 4h bar's range)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 12h EMA34 trend (prior completed 12h bar's EMA)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # Need at least 34 periods for EMA34
        return np.zeros(n)
    
    ema_34 = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    
    # Calculate 4h volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND price > 12h EMA34 (bullish bias) AND volume spike
            if (close[i] > high_ma[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short entry: price breaks below Donchian low AND price < 12h EMA34 (bearish bias) AND volume spike
            elif (close[i] < low_ma[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below Donchian low OR below 12h EMA34 (trend change)
            if close[i] < low_ma[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high OR above 12h EMA34 (trend change)
            if close[i] > high_ma[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
```

## Last Updated
2026-05-02 10:40
