# Strategy: 4h_Donchian20_1dEMA34_ATRVolConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.000 | +0.0% | 0.0% | 0 | FAIL |
| ETHUSDT | 0.065 | +20.2% | -18.9% | 87 | PASS |
| SOLUSDT | 0.955 | +206.3% | -28.5% | 104 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.809 | +23.6% | -8.1% | 26 | PASS |
| SOLUSDT | 0.515 | +17.4% | -10.0% | 29 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
- Long when price breaks above 20-period Donchian upper band AND close > 1d EMA34 (bullish trend)
- Short when price breaks below 20-period Donchian lower band AND close < 1d EMA34 (bearish trend)
- Volume must be > 2.0 * ATR(14) * close (volatility-adjusted volume filter)
- Exit on trend reversal (close crosses opposite EMA34) for faster mean reversion in chop
- Uses 4h primary timeframe with 1d HTF to target 75-200 trades over 4 years (19-50/year)
- Donchian channels provide clear breakout levels that work in trending markets
- 1d EMA34 ensures alignment with longer-term trend to avoid whipsaws in ranging/bear markets
- ATR-scaled volume filter adapts to changing volatility, reducing false breakouts
- Designed for BTC/ETH with edge in bull markets (breakout continuation) and bear markets (avoiding false breakouts via trend filter)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) using previous period (no look-ahead)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for dynamic volume threshold
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = np.nan
    tr3.iloc[0] = np.nan
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Dynamic volume threshold: volume > 2.0 * ATR * close (volatility-adjusted)
    vol_threshold = 2.0 * atr * close
    volume_confirm = volume > vol_threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band, trend up (close > EMA34), volume confirmation
            if close[i] > donchian_upper[i] and close[i] > ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band, trend down (close < EMA34), volume confirmation
            elif close[i] < donchian_lower[i] and close[i] < ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below 1d EMA34 (trend reversal)
            if close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above 1d EMA34 (trend reversal)
            if close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_ATRVolConfirm_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-24 03:24
