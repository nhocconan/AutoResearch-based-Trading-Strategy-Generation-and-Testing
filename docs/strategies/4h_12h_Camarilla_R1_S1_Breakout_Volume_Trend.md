# Strategy: 4h_12h_Camarilla_R1_S1_Breakout_Volume_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.182 | +28.0% | -9.8% | 272 | PASS |
| ETHUSDT | 0.035 | +21.3% | -14.5% | 258 | PASS |
| SOLUSDT | 0.441 | +54.1% | -25.1% | 213 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.330 | -5.2% | -6.6% | 104 | FAIL |
| ETHUSDT | 0.890 | +18.8% | -6.5% | 88 | PASS |
| SOLUSDT | 0.274 | +9.4% | -8.7% | 74 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_R1_S1_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for price action
    df_4h = get_htf_data(prices, '4h')
    
    # Get 12h data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 12h data for trend filter (EMA34)
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Camarilla pivot levels (R1, S1) on 12h timeframe
    def calculate_camarilla(high_arr, low_arr, close_arr):
        n_periods = len(close_arr)
        R1 = np.full(n_periods, np.nan)
        S1 = np.full(n_periods, np.nan)
        
        for i in range(1, n_periods):
            # Use previous period's OHLC
            high_prev = high_arr[i-1]
            low_prev = low_arr[i-1]
            close_prev = close_arr[i-1]
            
            # Camarilla formulas
            R1[i] = close_prev + (high_prev - low_prev) * 1.1 / 12
            S1[i] = close_prev - (high_prev - low_prev) * 1.1 / 12
        
        return R1, S1
    
    R1_12h, S1_12h = calculate_camarilla(high_12h, low_12h, close_12h)
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_12h, R1_12h)
    S1_aligned = align_htf_to_ltf(prices, df_12h, S1_12h)
    
    # Calculate volume spike indicator (volume > 2.0 * 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Ensure enough data for EMA34 (34) + 1 buffer
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(ema_34_12h_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume confirmation required
        vol_confirm = volume_spike[i]
        
        if position == 0:
            # Long when price breaks above R1 with volume AND 12h trend is up (price > EMA34)
            if close[i] > R1_aligned[i] and vol_confirm and close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S1 with volume AND 12h trend is down (price < EMA34)
            elif close[i] < S1_aligned[i] and vol_confirm and close[i] < ema_34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price falls below S1 (reversal) or 12h trend turns down
            if close[i] < S1_aligned[i] or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price rises above R1 (reversal) or 12h trend turns up
            if close[i] > R1_aligned[i] or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-19 16:24
