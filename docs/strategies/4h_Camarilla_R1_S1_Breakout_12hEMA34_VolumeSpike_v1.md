# Strategy: 4h_Camarilla_R1_S1_Breakout_12hEMA34_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.036 | +22.4% | -12.5% | 352 | PASS |
| ETHUSDT | 0.564 | +41.1% | -9.2% | 316 | PASS |
| SOLUSDT | -0.250 | +6.2% | -16.2% | 261 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.668 | -3.8% | -4.7% | 136 | FAIL |
| ETHUSDT | 1.429 | +21.7% | -4.8% | 121 | PASS |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA34 trend filter and volume spike confirmation.
# Long when price breaks above R1 AND close > 12h EMA34 AND volume > 2.0x 20-period average.
# Short when price breaks below S1 AND close < 12h EMA34 AND volume > 2.0x 20-period average.
# Exit on opposite breakout or ATR(14) trailing stop (2.0x).
# Uses 4h primary timeframe with 12h trend filter for noise reduction, targeting 75-200 trades over 4 years.
# Camarilla R1/S1 levels provide high-probability intraday support/resistance, EMA34 filters intermediate trend,
# volume confirms breakout authenticity. Designed to work in both bull and bear markets via strict entry conditions.

name = "4h_Camarilla_R1_S1_Breakout_12hEMA34_VolumeSpike_v1"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla pivot levels for 4h: based on previous bar's OHLC
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # Using previous bar to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # First bar: use current close
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    camarilla_range = prev_high - prev_low
    R1 = prev_close + 1.1 * camarilla_range / 12
    S1 = prev_close - 1.1 * camarilla_range / 12
    
    # Get 12h data for EMA34 trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h close
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF arrays to 4h timeframe (wait for completed 12h bar)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume filter: current 4h volume > 2.0x 20-period average (spike confirmation)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(50, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R1 AND close > 12h EMA34 AND volume spike
            if close[i] > R1[i] and close[i] > ema34_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: price breaks below S1 AND close < 12h EMA34 AND volume spike
            elif close[i] < S1[i] and close[i] < ema34_12h_aligned[i] and volume_filter[i]:
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
            # EXIT LONG: price breaks below S1 (opposite breakout) OR trailing stop hit
            breakout_exit = close[i] < S1[i]
            trailing_stop = close[i] < (highest_since_entry[i] - 2.0 * atr[i])
            if breakout_exit or trailing_stop:
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
            # EXIT SHORT: price breaks above R1 (opposite breakout) OR trailing stop hit
            breakout_exit = close[i] > R1[i]
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.0 * atr[i])
            if breakout_exit or trailing_stop:
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
2026-05-13 14:34
