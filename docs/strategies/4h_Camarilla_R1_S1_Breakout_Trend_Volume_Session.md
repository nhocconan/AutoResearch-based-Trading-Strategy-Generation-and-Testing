# Strategy: 4h_Camarilla_R1_S1_Breakout_Trend_Volume_Session

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.376 | +32.3% | -6.3% | 313 | PASS |
| ETHUSDT | 0.312 | +32.0% | -9.8% | 285 | PASS |
| SOLUSDT | 0.405 | +42.7% | -17.4% | 241 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.454 | -2.2% | -4.4% | 120 | FAIL |
| ETHUSDT | 0.609 | +12.2% | -4.0% | 111 | PASS |
| SOLUSDT | -0.084 | +5.2% | -6.5% | 88 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot (R1/S1) breakout with volume confirmation and 1d EMA trend filter
# Uses tighter R1/S1 levels for more frequent but still controlled entries (target: 25-40 trades/year)
# Trend filter prevents counter-trend trades in strong trends, volume confirms breakout strength
# Works in bull/bear via trend filter - only trades in direction of higher timeframe trend

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
    
    # Calculate Camarilla R1 and S1 from previous day (tighter than R2/S2)
    r1 = prev_close + (prev_range * 1.1 / 12)
    s1 = prev_close - (prev_range * 1.1 / 12)
    
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
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 + volume spike + uptrend (price > EMA34)
            if (close[i] > r1_aligned[i] and vol_spike[i] and close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + volume spike + downtrend (price < EMA34)
            elif (close[i] < s1_aligned[i] and vol_spike[i] and close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite S1/R1 level
            if position == 1:
                if close[i] < s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Trend_Volume_Session"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-22 08:40
