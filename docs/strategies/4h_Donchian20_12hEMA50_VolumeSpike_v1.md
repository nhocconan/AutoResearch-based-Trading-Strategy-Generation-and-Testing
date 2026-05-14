# Strategy: 4h_Donchian20_12hEMA50_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.197 | +7.1% | -18.7% | 90 | FAIL |
| ETHUSDT | 0.007 | +15.4% | -19.8% | 57 | PASS |
| SOLUSDT | 1.627 | +576.8% | -31.2% | 45 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.040 | +28.0% | -11.5% | 27 | PASS |
| SOLUSDT | 0.692 | +21.1% | -14.0% | 18 | PASS |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian upper band AND close > 12h EMA50 AND volume > 1.8x 20-period average.
# Short when price breaks below Donchian lower band AND close < 12h EMA50 AND volume > 1.8x 20-period average.
# Exit on ATR(14) trailing stop (2.5x) or opposite breakout.
# Uses 4h primary timeframe with 12h trend filter for noise reduction, targeting 75-200 total trades over 4 years.
# Donchian channels provide clear trend-following structure, 12h EMA50 filters medium-term trend,
# volume confirms breakout authenticity. Designed to work in both bull and bear markets via strict entry conditions.

name = "4h_Donchian20_12hEMA50_VolumeSpike_v1"
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
    
    # Calculate Donchian channels (20-period) using previous period's data to avoid look-ahead
    # Upper band = highest high of previous 20 periods
    # Lower band = lowest low of previous 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Shift by 1 to use only completed periods (avoid look-ahead)
    upper_band = np.roll(highest_high, 1)
    lower_band = np.roll(lowest_low, 1)
    # First bar uses current values (no previous data)
    upper_band[0] = highest_high[0]
    lower_band[0] = lowest_low[0]
    
    # Get 12h data for EMA50 trend filter (MTF)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h close
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF arrays to 4h timeframe (wait for completed 12h bar)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: current 4h volume > 1.8x 20-period average (spike confirmation)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above upper band AND close > 12h EMA50 AND volume spike
            if close[i] > upper_band[i] and close[i] > ema50_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: price breaks below lower band AND close < 12h EMA50 AND volume spike
            elif close[i] < lower_band[i] and close[i] < ema50_12h_aligned[i] and volume_filter[i]:
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
            # EXIT LONG: trailing stop hit (opposite breakout handled by next bar's entry logic)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.5 * atr[i])
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
            # EXIT SHORT: trailing stop hit (opposite breakout handled by next bar's entry logic)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.5 * atr[i])
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
2026-05-13 14:42
