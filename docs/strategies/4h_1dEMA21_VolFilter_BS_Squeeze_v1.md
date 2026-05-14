# Strategy: 4h_1dEMA21_VolFilter_BS_Squeeze_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.001 | +16.9% | -19.8% | 245 | PASS |
| ETHUSDT | 0.390 | +52.6% | -17.1% | 264 | PASS |
| SOLUSDT | 1.082 | +266.0% | -33.0% | 289 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.816 | -5.5% | -12.3% | 93 | FAIL |
| ETHUSDT | 0.237 | +9.7% | -11.3% | 96 | PASS |
| SOLUSDT | 0.453 | +15.9% | -10.1% | 84 | PASS |

## Code
```python
#!/usr/bin/env python3
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA(21) for trend filter
    ema_21_1d = pd.Series(df_1d['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d Bollinger Band width for squeeze detection
    sma_20_1d = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + (2 * std_20_1d)
    lower_bb_1d = sma_20_1d - (2 * std_20_1d)
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma_20_1d
    bb_width_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_width_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_21_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(bb_width_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: price > 1d EMA21 for long, price < 1d EMA21 for short
        trend_filter_long = price > ema_21_1d_aligned[i]
        trend_filter_short = price < ema_21_1d_aligned[i]
        
        # Volatility filter: 1d ATR > 2% of price to avoid low volatility periods
        vol_filter = atr_1d_aligned[i] / price > 0.02 if price > 0 else False
        
        # Bollinger Band squeeze detection: bandwidth < 4%
        bb_squeeze = bb_width_1d_aligned[i] < 0.04
        
        if position == 0:
            # Long setup: price above 1d EMA21 + volatility filter + not in squeeze
            if trend_filter_long and vol_filter and not bb_squeeze:
                position = 1
                signals[i] = position_size
            # Short setup: price below 1d EMA21 + volatility filter + not in squeeze
            elif trend_filter_short and vol_filter and not bb_squeeze:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 1d EMA21
            if price < ema_21_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 1d EMA21
            if price > ema_21_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1dEMA21_VolFilter_BS_Squeeze_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-14 05:54
