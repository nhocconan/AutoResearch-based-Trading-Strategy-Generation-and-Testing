# Strategy: 4h_WilliamsR_Extreme_1dEMA50_Trend_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.180 | +14.2% | -10.1% | 193 | FAIL |
| ETHUSDT | 0.147 | +26.7% | -11.1% | 164 | PASS |
| SOLUSDT | 0.505 | +58.9% | -13.4% | 142 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.024 | +5.9% | -7.7% | 64 | PASS |
| SOLUSDT | -0.177 | +3.4% | -8.0% | 49 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Extreme + 1d EMA50 Trend Filter + Volume Spike.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when Williams %R(14) crosses above -20 (oversold bounce) AND price > 1d EMA50 AND volume > 1.5 * 4h volume MA(20);
         Short when Williams %R(14) crosses below -80 (overbought rejection) AND price < 1d EMA50 AND volume > 1.5 * 4h volume MA(20).
- Exit: Long exits when Williams %R(14) crosses below -50; Short exits when Williams %R(14) crosses above -50.
- Signal size: 0.25 discrete to balance capture and fee control.
- Williams %R identifies overextended moves likely to reverse; volume confirms participation; EMA50 ensures trend alignment.
- Works in bull (buying oversold bounces in uptrend) and bear (selling overbought rejections in downtrend).
- Avoids counter-trend trades and whipsaws via strict volume and trend filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA50 for 1d
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Williams %R(14) on 4h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        -100 * (highest_high - close) / (highest_high - lowest_low),
        -50  # neutral when range is zero
    )
    
    # Get 4h data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14, 20)  # EMA50 needs 50, Williams %R needs 14, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_volume = volume[i]
        prev_williams_r = williams_r[i-1] if i > 0 else -50
        
        # Volume confirmation: 1.5x threshold
        vol_confirm = curr_volume > 1.5 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Williams %R crosses above -20 (oversold bounce) AND price > 1d EMA50 (uptrend)
                if prev_williams_r <= -20 and curr_williams_r > -20 and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -80 (overbought rejection) AND price < 1d EMA50 (downtrend)
                elif prev_williams_r >= -80 and curr_williams_r < -80 and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when Williams %R crosses below -50
            if prev_williams_r >= -50 and curr_williams_r < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when Williams %R crosses above -50
            if prev_williams_r <= -50 and curr_williams_r > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-24 17:40
