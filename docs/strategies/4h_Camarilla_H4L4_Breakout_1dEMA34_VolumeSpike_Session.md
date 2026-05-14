# Strategy: 4h_Camarilla_H4L4_Breakout_1dEMA34_VolumeSpike_Session

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.142 | +25.1% | -7.7% | 247 | PASS |
| ETHUSDT | 0.264 | +29.7% | -8.1% | 215 | PASS |
| SOLUSDT | -0.065 | +16.4% | -18.3% | 201 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.750 | -3.0% | -4.4% | 95 | FAIL |
| ETHUSDT | 0.035 | +6.3% | -5.0% | 91 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla H4L4 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above H4 level AND 1d close > 1d EMA34 AND 4h volume > 2.0x 20-period volume MA.
# Short when price breaks below L4 level AND 1d close < 1d EMA34 AND 4h volume > 2.0x 20-period volume MA.
# Exit when price retests the broken level (H4 for longs, L4 for shorts) or trend changes.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size 0.25.
# Designed for 4h timeframe to achieve 75-200 total trades over 4 years (19-50/year) with strict entry conditions.
# Camarilla pivot levels provide statistically significant support/resistance, 1d EMA34 filters for higher-timeframe alignment, volume confirms institutional participation.
# Works in both bull and bear markets by only trading breakouts in the direction of the 1d trend when volume confirms.

name = "4h_Camarilla_H4L4_Breakout_1dEMA34_VolumeSpike_Session"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d
    # H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # where C, H, L are from previous 1d candle
    df_1d_close = df_1d['close'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    
    # Previous day's values (for today's Camarilla levels)
    prev_close = np.roll(df_1d_close, 1)
    prev_high = np.roll(df_1d_high, 1)
    prev_low = np.roll(df_1d_low, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla H4 and L4 levels
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 4h volume 20-period MA for spike detection
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(volume_ma_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        h4_level = camarilla_h4_aligned[i]
        l4_level = camarilla_l4_aligned[i]
        
        # Volume spike condition: current 4h volume > 2.0x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_20[i] * 2.0)
        
        # 1d trend conditions
        trend_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: Price breaks above H4 AND 1d uptrend AND volume spike AND session
            if close_val > h4_level and trend_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below L4 AND 1d downtrend AND volume spike AND session
            elif close_val < l4_level and trend_down and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price retests H4 level OR trend changes to down
            if close_val < h4_level or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price retests L4 level OR trend changes to up
            if close_val > l4_level or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-03 11:26
