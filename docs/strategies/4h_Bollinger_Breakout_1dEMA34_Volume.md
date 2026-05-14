# Strategy: 4h_Bollinger_Breakout_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.372 | +29.7% | -2.6% | 169 | PASS |
| ETHUSDT | 0.266 | +28.6% | -4.2% | 148 | PASS |
| SOLUSDT | 0.146 | +26.8% | -11.7% | 140 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.855 | -1.4% | -3.6% | 59 | FAIL |
| ETHUSDT | 0.083 | +6.7% | -3.9% | 59 | PASS |
| SOLUSDT | -0.425 | +3.1% | -5.6% | 44 | FAIL |

## Code
```python
#!/usr/bin/env python3

"""
Hypothesis: 4-hour Bollinger Band breakout with 1-day trend filter and volume confirmation.
Bollinger Bands provide volatility-based support/resistance levels. The daily trend filter
ensures trades align with the higher timeframe direction, reducing counter-trend trades.
Volume spikes confirm institutional participation. This combination should work in both
bull and bear markets by adapting to the daily trend while capturing mean reversion
breakouts from volatility contractions. Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA for trend filter (34-period)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily EMA to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Bollinger Bands (20, 2.0)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2.0 * std_20
    lower_band = sma_20 - 2.0 * std_20
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper band, above daily EMA, volume spike
            if (close[i] > upper_band[i] and    # Break above upper BB
                close[i] > ema_34_aligned[i] and # Above daily EMA (bullish trend)
                volume[i] > 2.0 * vol_avg_20[i]): # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band, below daily EMA, volume spike
            elif (close[i] < lower_band[i] and   # Break below lower BB
                  close[i] < ema_34_aligned[i] and # Below daily EMA (bearish trend)
                  volume[i] > 2.0 * vol_avg_20[i]): # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to middle (SMA20) or opposite band touch
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below SMA20
                if close[i] < sma_20[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above SMA20
                if close[i] > sma_20[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Bollinger_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-22 17:47
