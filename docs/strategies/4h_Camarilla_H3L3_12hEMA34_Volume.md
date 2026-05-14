# Strategy: 4h_Camarilla_H3L3_12hEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.427 | -1.3% | -18.3% | 238 | FAIL |
| ETHUSDT | 0.385 | +45.6% | -13.3% | 222 | PASS |
| SOLUSDT | 0.741 | +111.3% | -23.0% | 196 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.456 | +13.4% | -11.6% | 76 | PASS |
| SOLUSDT | 0.023 | +5.4% | -12.8% | 67 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_H3L3_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for EMA34 filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA34 on 12h close
    close_12h = df_12h['close']
    ema_34_12h = close_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Load 1d data for Camarilla pivot levels (H3/L3)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels (H3, L3) from previous daily bar
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    H3 = pivot + (range_hl * 1.1 / 4)
    L3 = pivot - (range_hl * 1.1 / 4)
    
    # Align H3/L3 to 4h (wait for daily close)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume filter: current volume > 1.8 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        H3_val = H3_aligned[i]
        L3_val = L3_aligned[i]
        ema_12h_val = ema_34_12h_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: break above H3 with volume confirmation and 12h EMA34 above price (bullish bias)
            if close_val > H3_val and vol_filter and (close_val > ema_12h_val):
                signals[i] = 0.25
                position = 1
            # Short: break below L3 with volume confirmation and 12h EMA34 below price (bearish bias)
            elif close_val < L3_val and vol_filter and (close_val < ema_12h_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below L3 or 12h EMA34 turns bearish
            if close_val < L3_val or (close_val < ema_12h_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above H3 or 12h EMA34 turns bullish
            if close_val > H3_val or (close_val > ema_12h_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-18 23:06
