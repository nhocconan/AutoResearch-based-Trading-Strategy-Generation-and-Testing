# Strategy: 12h_PreviousDayBreakout_EMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.070 | +18.6% | -6.4% | 92 | FAIL |
| ETHUSDT | 0.141 | +26.4% | -6.9% | 80 | PASS |
| SOLUSDT | 0.132 | +26.2% | -19.7% | 79 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.115 | +7.1% | -5.4% | 34 | PASS |
| SOLUSDT | -0.722 | -3.1% | -12.5% | 31 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h price action based on 1d OHLC ranges with volume confirmation and trend filter
# Long when price exceeds previous day's high + volume spike + price > 1d EMA34
# Short when price falls below previous day's low + volume spike + price < 1d EMA34
# Exit when price returns to previous day's close or trend reverses
# Designed for low trade frequency (<30/year) with potential edge in trending markets
# Uses simple price levels that work across market regimes

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Load 1d data for price levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's OHLC for breakout levels
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    # Align price levels to 12h timeframe
    high_prev_aligned = align_htf_to_ltf(prices, df_1d, high_1d_prev)
    low_prev_aligned = align_htf_to_ltf(prices, df_1d, low_1d_prev)
    close_prev_aligned = align_htf_to_ltf(prices, df_1d, close_1d_prev)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume spike using 20-period average
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(high_prev_aligned[i]) or 
            np.isnan(low_prev_aligned[i]) or 
            np.isnan(close_prev_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        high_prev = high_prev_aligned[i]
        low_prev = low_prev_aligned[i]
        close_prev = close_prev_aligned[i]
        ema_val = ema_34_aligned[i]
        
        # Volume filter: current volume > 1.8 * 20-day average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above previous day's high + uptrend + volume spike
            if price > high_prev and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below previous day's low + downtrend + volume spike
            elif price < low_prev and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to previous day's close or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to previous day's close or trend turns down
                if price <= close_prev or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to previous day's close or trend turns up
                if price >= close_prev or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_PreviousDayBreakout_EMA34_Volume"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-22 02:59
