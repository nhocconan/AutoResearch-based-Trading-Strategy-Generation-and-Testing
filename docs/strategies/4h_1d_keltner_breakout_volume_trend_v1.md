# Strategy: 4h_1d_keltner_breakout_volume_trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.212 | +29.4% | -10.2% | 104 | PASS |
| ETHUSDT | 0.304 | +36.2% | -10.5% | 96 | PASS |
| SOLUSDT | 0.742 | +91.5% | -16.8% | 91 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.605 | +1.0% | -5.6% | 38 | FAIL |
| ETHUSDT | 0.631 | +15.3% | -7.6% | 36 | PASS |
| SOLUSDT | 0.236 | +8.9% | -10.6% | 33 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_1d_keltner_breakout_volume_trend_v1
Strategy: 4h Keltner channel breakout with volume confirmation and 1d EMA trend filter
Timeframe: 4h
Leverage: 1.0
Hypothesis: Combines Keltner channel breakouts (ATR-based volatility channels) for volatility breakouts with volume confirmation (>2.0x average volume) and filtered by 1d EMA50 trend alignment. Keltner channels adapt to volatility better than fixed percentage bands, reducing false breakouts in low volatility regimes. Designed to work in both bull and bear markets by following the higher timeframe trend. Target: 20-50 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_keltner_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # ATR for Keltner channels (20-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Keltner channels (20-period EMA middle, ATR multiplier)
    ema_middle = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper = ema_middle + (2.0 * atr)
    keltner_lower = ema_middle - (2.0 * atr)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_middle[i]) or np.isnan(keltner_upper[i]) or 
            np.isnan(keltner_lower[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Breakout conditions using Keltner channels
        breakout_up = price_close > keltner_upper[i]
        breakout_down = price_close < keltner_lower[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: upward breakout with volume in uptrend
        long_signal = breakout_up and vol_confirmed and uptrend_1d
        
        # Short: downward breakout with volume in downtrend
        short_signal = breakout_down and vol_confirmed and downtrend_1d
        
        # Exit when price returns to the middle line (EMA20)
        exit_long = position == 1 and price_close < ema_middle[i]
        exit_short = position == -1 and price_close > ema_middle[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
```

## Last Updated
2026-04-11 13:04
