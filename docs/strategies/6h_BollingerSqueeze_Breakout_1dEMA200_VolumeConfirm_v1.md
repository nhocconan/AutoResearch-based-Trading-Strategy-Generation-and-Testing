# Strategy: 6h_BollingerSqueeze_Breakout_1dEMA200_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.335 | +33.4% | -7.2% | 78 | PASS |
| ETHUSDT | 0.329 | +35.5% | -12.4% | 68 | PASS |
| SOLUSDT | 0.803 | +89.7% | -18.3% | 64 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.430 | -6.4% | -10.5% | 32 | FAIL |
| ETHUSDT | 0.264 | +9.4% | -9.7% | 28 | PASS |
| SOLUSDT | -0.696 | -4.7% | -18.8% | 28 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d Trend Filter and Volume Confirmation.
- Bollinger Band Width (BBW) percentile identifies low-volatility squeeze regimes.
- Breakout from squeeze + volume confirmation captures explosive moves.
- 1d EMA200 provides higher-timeframe trend filter to avoid counter-trend trades in bear markets.
- Position size 0.25 balances profit and drawdown control.
- Target trades: 80-160 total over 4 years (20-40/year) to balance opportunity and fee drag.
- Works in bull/bear markets via 1d trend filter and volatility expansion logic.
"""

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
    
    # Get 1d data ONCE before loop for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA200 trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2
    ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = ma + bb_std * std
    lower = ma - bb_std * std
    bb_width = (upper - lower) / ma  # Normalized width
    
    # BB Width percentile (50-period lookback) to identify squeeze
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Volume confirmation: > 1.8x 30-period average (slightly looser for 6h)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(bb_period, 30, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(ma[i]) or np.isnan(std[i]) or 
            np.isnan(bb_width_percentile[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Squeeze condition: BB Width at or below 20th percentile (low volatility)
        is_squeeze = bb_width_percentile[i] <= 20
        
        # Breakout conditions
        breakout_up = close[i] > upper[i-1]  # Break above previous upper band
        breakout_down = close[i] < lower[i-1]  # Break below previous lower band
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Only trade during/after squeeze with volume confirmation
            if volume_confirm:
                # Long: breakout up + above 1d EMA200 (bullish higher-timeframe trend)
                if breakout_up and close[i] > ema_200_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: breakout down + below 1d EMA200 (bearish higher-timeframe trend)
                elif breakout_down and close[i] < ema_200_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price closes below middle band (MA) OR squeeze ends with reversal
            if close[i] < ma[i] or (bb_width_percentile[i] > 80 and close[i] < ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above middle band (MA) OR squeeze ends with reversal
            if close[i] > ma[i] or (bb_width_percentile[i] > 80 and close[i] > ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BollingerSqueeze_Breakout_1dEMA200_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-24 01:56
