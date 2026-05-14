# Strategy: 6h_WilliamsAlligator_1dEMA50_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.227 | +10.0% | -14.4% | 156 | FAIL |
| ETHUSDT | 0.053 | +21.8% | -12.0% | 140 | PASS |
| SOLUSDT | 0.511 | +68.4% | -21.1% | 148 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.248 | +9.4% | -11.4% | 50 | PASS |
| SOLUSDT | 0.035 | +5.7% | -14.1% | 47 | PASS |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator + 1d trend filter + volume confirmation.
# Long when Alligator jaws < teeth < lips (bullish alignment) AND price > lips AND close > 1d EMA50 AND volume > 1.5x 20-period average
# Short when Alligator jaws > teeth > lips (bearish alignment) AND price < lips AND close < 1d EMA50 AND volume > 1.5x 20-period average
# Exit on opposite alignment (jaws-teeth-lips crossed) or ATR trailing stop (2.0x)
# Uses 6h timeframe with 1d trend filter for noise reduction, targeting 50-150 trades over 4 years.
# Williams Alligator identifies trend structure via SMAs, EMA50 filters intermediate trend, volume confirms authenticity.

name = "6h_WilliamsAlligator_1dEMA50_VolumeSpike_v1"
timeframe = "6h"
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
    
    # Williams Alligator on 6h: SMA(13,8), SMA(8,5), SMA(5,3) of median price
    median_price = (high + low) / 2
    def sma(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).mean().values
    
    jaws = sma(median_price, 13)  # SMA(13,8) -> blue line
    teeth = sma(median_price, 8)  # SMA(8,5)   -> red line
    lips = sma(median_price, 5)   # SMA(5,3)   -> green line
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d close
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF arrays to 6h timeframe (wait for completed 1d bar)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current 6h volume > 1.5x 20-period average (spike confirmation)
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(50, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        # Bullish alignment: jaws < teeth < lips
        bullish_align = jaws[i] < teeth[i] and teeth[i] < lips[i]
        # Bearish alignment: jaws > teeth > lips
        bearish_align = jaws[i] > teeth[i] and teeth[i] > lips[i]
        
        if position == 0:
            # LONG: bullish alignment AND price > lips AND close > 1d EMA50 AND volume spike
            if bullish_align and close[i] > lips[i] and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: bearish alignment AND price < lips AND close < 1d EMA50 AND volume spike
            elif bearish_align and close[i] < lips[i] and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
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
            # EXIT LONG: bearish alignment (jaws > teeth > lips) OR trailing stop hit
            alignment_exit = bearish_align
            trailing_stop = close[i] < (highest_since_entry[i] - 2.0 * atr[i])
            if alignment_exit or trailing_stop:
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
            # EXIT SHORT: bullish alignment (jaws < teeth < lips) OR trailing stop hit
            alignment_exit = bullish_align
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.0 * atr[i])
            if alignment_exit or trailing_stop:
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
2026-05-13 14:29
