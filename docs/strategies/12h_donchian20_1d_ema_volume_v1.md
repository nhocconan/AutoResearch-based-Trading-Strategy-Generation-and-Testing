# Strategy: 12h_donchian20_1d_ema_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.741 | -15.6% | -18.9% | 453 | FAIL |
| ETHUSDT | -0.902 | -31.1% | -39.2% | 475 | FAIL |
| SOLUSDT | 0.201 | +31.8% | -30.2% | 457 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.147 | +7.6% | -13.7% | 146 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with 1-day EMA(50) trend filter and volume confirmation
# Designed for low frequency (target: 12-37 trades/year) to minimize fee drag
# Works in both bull and bear markets by aligning with higher timeframe trend (1d EMA)
# Donchian breakouts capture strong momentum moves; EMA filter avoids counter-trend trades
# Volume confirmation ensures breakouts have institutional participation

name = "12h_donchian20_1d_ema_volume_v1"
timezone = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i-1] if i > 0 else False
        breakout_down = close[i] < donchian_low[i-1] if i > 0 else False
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit on downside breakout or trend reversal
            if breakout_down or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on upside breakout or trend reversal
            if breakout_up or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Entry conditions with trend and volume confirmation
            # Long on upside breakout in uptrend
            if breakout_up and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short on downside breakout in downtrend
            elif breakout_down and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 05:53
