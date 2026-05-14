# Strategy: 12h_Camarilla_H3L3_Breakout_1dEMA34_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.255 | +12.4% | -5.7% | 70 | FAIL |
| ETHUSDT | 0.050 | +22.4% | -7.0% | 63 | PASS |
| SOLUSDT | 0.258 | +36.9% | -24.1% | 64 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.520 | +13.0% | -5.9% | 24 | PASS |
| SOLUSDT | -1.157 | -8.7% | -14.4% | 24 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 Breakout with 1d Trend Filter and Volume Confirmation.
- Uses Camarilla pivot levels (H3, L3) from prior 1d for high-probability breakout levels.
- Breakout above H3 or below L3 with volume confirmation captures institutional moves.
- 1d EMA34 provides higher-timeframe trend filter to align with medium-term momentum.
- Position size 0.25 balances profit and drawdown control.
- Target trades: 50-150 total over 4 years (12-37/year) to minimize fee drag.
- Works in bull/bear markets via 1d trend filter and volatility-based logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla pivots and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d candle (H3, L3, H4, L4)
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    #            L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * range_1d * 1.1 / 4
    camarilla_l3 = close_1d - 1.1 * range_1d * 1.1 / 4
    camarilla_h4 = close_1d + 1.1 * range_1d * 1.1 / 2
    camarilla_l4 = close_1d - 1.1 * range_1d * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: > 2.0x 24-period average (strict for 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(h3_1d_aligned[i]) or 
            np.isnan(l3_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Only trade with volume confirmation
            if volume_confirm:
                # Long: break above H3 + above 1d EMA34 (bullish higher-timeframe trend)
                if close[i] > h3_1d_aligned[i] and close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: break below L3 + below 1d EMA34 (bearish higher-timeframe trend)
                elif close[i] < l3_1d_aligned[i] and close[i] < ema_34_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price closes below L3 (reversal) OR below EMA34 (trend change)
            if close[i] < l3_1d_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above H3 (reversal) OR above EMA34 (trend change)
            if close[i] > h3_1d_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-24 02:00
