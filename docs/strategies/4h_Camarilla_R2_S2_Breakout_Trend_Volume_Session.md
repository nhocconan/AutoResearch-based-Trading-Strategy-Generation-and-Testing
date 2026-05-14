# Strategy: 4h_Camarilla_R2_S2_Breakout_Trend_Volume_Session

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.488 | +35.5% | -6.3% | 300 | PASS |
| ETHUSDT | 0.399 | +35.2% | -9.9% | 275 | PASS |
| SOLUSDT | 0.308 | +36.6% | -17.4% | 230 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.614 | -2.9% | -4.4% | 116 | FAIL |
| ETHUSDT | 0.359 | +9.5% | -4.7% | 110 | PASS |
| SOLUSDT | -0.113 | +4.9% | -6.8% | 83 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot (R2/S2) breakout with volume confirmation and trend filter
# Uses 1d EMA34 for trend direction to avoid counter-trend trades
# Target: 25-40 trades/year per symbol, works in bull/bear via trend filter
# Breakout at R2/S2 provides stronger signal than R1/S1 with fewer false signals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Camarilla pivot and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily range
    daily_range = high_1d - low_1d
    
    # Calculate Camarilla levels for previous day
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_range = np.roll(daily_range, 1)
    
    # Set first day values to NaN
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_range[0] = np.nan
    
    # Calculate Camarilla R2 and S2 from previous day
    r2 = prev_close + (prev_range * 1.1 / 6)
    s2 = prev_close - (prev_range * 1.1 / 6)
    
    # Calculate 34-period EMA on daily close for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34 = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume spike filter (20-period on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to 4-hour timeframe
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R2 + volume spike + uptrend (price > EMA34)
            if (close[i] > r2_aligned[i] and vol_spike[i] and close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S2 + volume spike + downtrend (price < EMA34)
            elif (close[i] < s2_aligned[i] and vol_spike[i] and close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite S2/R2 level
            if position == 1:
                if close[i] < s2_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > r2_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R2_S2_Breakout_Trend_Volume_Session"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-22 08:30
