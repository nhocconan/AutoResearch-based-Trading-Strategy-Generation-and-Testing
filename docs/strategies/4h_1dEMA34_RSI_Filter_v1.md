# Strategy: 4h_1dEMA34_RSI_Filter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.317 | +39.1% | -10.7% | 189 | PASS |
| ETHUSDT | 0.177 | +30.0% | -15.3% | 233 | PASS |
| SOLUSDT | 0.062 | +15.9% | -33.6% | 270 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.712 | -3.2% | -8.2% | 83 | FAIL |
| ETHUSDT | 0.656 | +18.5% | -9.1% | 63 | PASS |
| SOLUSDT | 0.784 | +24.1% | -11.8% | 66 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d RSI(14) for momentum filter
    delta = np.diff(df_1d['close'], prepend=df_1d['close'].iloc[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: price > 1d EMA34 for long, price < 1d EMA34 for short
        trend_filter_long = price > ema_34_1d_aligned[i]
        trend_filter_short = price < ema_34_1d_aligned[i]
        
        # Momentum filter: RSI between 35 and 65 to avoid extremes
        mom_filter = (rsi_1d_aligned[i] >= 35) & (rsi_1d_aligned[i] <= 65)
        
        if position == 0:
            # Long setup: price above 1d EMA34 + momentum filter
            if trend_filter_long and mom_filter:
                position = 1
                signals[i] = position_size
            # Short setup: price below 1d EMA34 + momentum filter
            elif trend_filter_short and mom_filter:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 1d EMA34 OR RSI > 70 (overbought)
            if price < ema_34_1d_aligned[i] or rsi_1d_aligned[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 1d EMA34 OR RSI < 30 (oversold)
            if price > ema_34_1d_aligned[i] or rsi_1d_aligned[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1dEMA34_RSI_Filter_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-14 06:13
