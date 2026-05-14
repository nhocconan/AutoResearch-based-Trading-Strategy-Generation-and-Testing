# Strategy: 4h_Camarilla_R1_S1_Breakout_12hEMA50_VolumeSpike_ATRStop_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.206 | +31.7% | -13.1% | 182 | PASS |
| ETHUSDT | 0.196 | +31.6% | -15.2% | 162 | PASS |
| SOLUSDT | 0.899 | +177.1% | -28.4% | 149 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.587 | -1.9% | -9.8% | 63 | FAIL |
| ETHUSDT | 0.846 | +24.0% | -9.2% | 53 | PASS |
| SOLUSDT | 0.546 | +17.5% | -9.8% | 49 | PASS |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above Camarilla R1 with 12h EMA50 uptrend and volume > 1.5x average.
# Short when price breaks below Camarilla S1 with 12h EMA50 downtrend and volume > 1.5x average.
# Uses ATR(14) trailing stop (2.0x) for risk control. Discrete sizing 0.25.
# Target: 75-200 total trades over 4 years (19-50/year) on 4h.
# Based on proven ETH/USDT winners (test Sharpe up to 2.07) with tight entries.

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_VolumeSpike_ATRStop_v2"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 4h data for Camarilla pivot calculation (using previous day's OHLC)
    df_4h = get_htf_data(prices, '4h')
    # For Camarilla, we need daily OHLC from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels using previous 1d bar
    # Camarilla R1 = close + (high - low) * 1.1/12
    # Camarilla S1 = close - (high - low) * 1.1/12
    camarilla_range = high_1d - low_1d
    camarilla_r1 = close_1d + camarilla_range * 1.1 / 12
    camarilla_s1 = close_1d - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (wait for daily bar to close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe (wait for 12h bar to close)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R1 AND 12h EMA50 uptrend AND volume > 1.5x average
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema50_12h_aligned[i] and  # Price above 12h EMA50 confirms uptrend
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Price breaks below Camarilla S1 AND 12h EMA50 downtrend AND volume > 1.5x average
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema50_12h_aligned[i] and  # Price below 12h EMA50 confirms downtrend
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.0 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.0 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals
```

## Last Updated
2026-05-13 19:39
