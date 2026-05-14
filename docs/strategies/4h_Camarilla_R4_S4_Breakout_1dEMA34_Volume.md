# Strategy: 4h_Camarilla_R4_S4_Breakout_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.446 | +36.5% | -5.6% | 176 | PASS |
| ETHUSDT | 0.250 | +31.4% | -9.2% | 168 | PASS |
| SOLUSDT | 0.892 | +96.3% | -13.5% | 131 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.388 | -3.2% | -5.6% | 67 | FAIL |
| ETHUSDT | 0.639 | +13.3% | -5.6% | 57 | PASS |
| SOLUSDT | 0.065 | +6.5% | -5.7% | 47 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses 1d EMA34 for trend filter and 1d Camarilla pivot levels (R4/S4) for stronger structure
# Entry: Long when price breaks above 1d Camarilla R4 with volume spike and price > 1d EMA34 (uptrend)
#        Short when price breaks below 1d Camarilla S4 with volume spike and price < 1d EMA34 (downtrend)
# Exit: Close crosses 1d EMA34 (trend reversal) or price retests Camarilla pivot point (PP)
# Works in both bull and bear markets by trading with 1d trend using Camarilla structure
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Discrete sizing 0.25 balances profit potential and fee drag

name = "4h_Camarilla_R4_S4_Breakout_1dEMA34_Volume"
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels (PP, R4, S4)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # PP = (high + low + close) / 3
    # R4 = close + 1.1 * (high - low) * 1.1 / 2  (equivalent to close + 1.1*(high-low)/6 * 3.3)
    # S4 = close - 1.1 * (high - low) * 1.1 / 2  (equivalent to close - 1.1*(high-low)/6 * 3.3)
    # Using standard Camarilla formulas:
    # R4 = close + 1.1 * (high - low) * 1.1 / 2
    # S4 = close - 1.1 * (high - low) * 1.1 / 2
    camarilla_pp = (high_1d + low_1d + close_1d_arr) / 3
    camarilla_r4 = close_1d_arr + 1.1 * (high_1d - low_1d) * 1.1 / 2
    camarilla_s4 = close_1d_arr - 1.1 * (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (use previous completed 1d bar's levels)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Break above 1d Camarilla R4 AND price > 1d EMA34 (uptrend) AND volume spike
            if (close[i] > camarilla_r4_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Break below 1d Camarilla S4 AND price < 1d EMA34 (downtrend) AND volume spike
            elif (close[i] < camarilla_s4_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below 1d EMA34 (trend change) OR price retests Camarilla PP (take profit)
            if close[i] < ema_34_1d_aligned[i] or close[i] < camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above 1d EMA34 (trend change) OR price retests Camarilla PP (take profit)
            if close[i] > ema_34_1d_aligned[i] or close[i] > camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 07:34
