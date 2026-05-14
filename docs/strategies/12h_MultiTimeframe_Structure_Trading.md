# Strategy: 12h_MultiTimeframe_Structure_Trading

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.297 | -2.3% | -24.3% | 238 | FAIL |
| ETHUSDT | 0.337 | +46.9% | -17.4% | 234 | PASS |
| SOLUSDT | 0.714 | +139.0% | -32.3% | 270 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.341 | +12.4% | -10.7% | 78 | PASS |
| SOLUSDT | 0.113 | +6.4% | -12.6% | 76 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
12h_MultiTimeframe_Structure_Trading
Combines 1d structure with 12h momentum for high-probability entries.
- Primary signal: Price breaks above/below 1d Donchian(20) channels
- Trend filter: 12h price above/below 12h EMA34 (trend alignment)
- Entry filter: Volume > 1.3x 20-period average + momentum confirmation (RSI > 50 for long, < 50 for short)
- Exit: Opposite signal or trend reversal
- Designed for 15-25 trades/year per symbol (60-100 total over 4 years)
Works in bull markets (breakout continuation) and bear markets (breakdown continuation)
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
    
    # Get 1d data for structure (Donchian channels)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Get 12h data for trend filter (EMA34)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 12h RSI for momentum confirmation
    delta = pd.Series(close_12h).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi.values)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need 20 for Donchian/volume MA + buffer
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 12h price relative to EMA34
        bull_trend = ema_34_12h_aligned[i] > 0 and close_12h[i // 12] > ema_34_12h[i // 12] if i >= 12 else False
        bear_trend = ema_34_12h_aligned[i] > 0 and close_12h[i // 12] < ema_34_12h[i // 12] if i >= 12 else False
        
        # Simplified trend check using current price vs EMA (more reliable)
        bull_trend = close[i] > ema_34_12h_aligned[i]
        bear_trend = close[i] < ema_34_12h_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high_aligned[i]
        breakdown_down = close[i] < donchian_low_aligned[i]
        
        # Entry filters
        volume_ok = volume[i] > 1.3 * vol_ma[i]
        mom_long = rsi_aligned[i] > 50
        mom_short = rsi_aligned[i] < 50
        
        if position == 0:
            # Long: bull trend + breakout up + volume + momentum
            if bull_trend and breakout_up and volume_ok and mom_long:
                signals[i] = 0.25
                position = 1
            # Short: bear trend + breakdown down + volume + momentum
            elif bear_trend and breakdown_down and volume_ok and mom_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change or breakdown
            if not bull_trend or breakdown_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change or breakout
            if not bear_trend or breakout_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_MultiTimeframe_Structure_Trading"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-18 13:23
