# Strategy: 4h_1d_camarilla_breakout_vol_chop_v4

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.151 | +15.6% | -8.3% | 284 | FAIL |
| ETHUSDT | 0.434 | +39.7% | -6.8% | 270 | PASS |
| SOLUSDT | 0.402 | +42.9% | -13.8% | 195 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.189 | +7.8% | -10.5% | 94 | PASS |
| SOLUSDT | -0.490 | +0.7% | -9.2% | 77 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
    # Camarilla levels from 1d provide institutional support/resistance
    # Break above H3 or below L3 with volume > 2x 20-period MA signals institutional participation
    # Chop > 61.8 ensures we only trade in ranging markets where mean reversion works
    # Discrete position sizing (0.25) to minimize fee churn. Target: 20-50 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H3 = close + 1.1*(high-low)/4
    # L3 = close - 1.1*(high-low)/4
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align to 4h timeframe (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 2x 20-period MA
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    vol_ratio = np.full(n, np.nan)
    for i in range(20, n):
        if vol_ma_20[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma_20[i]
        else:
            vol_ratio[i] = 1.0
    
    # Chop regime filter: CHOP > 61.8 = ranging market (good for mean reversion)
    # Calculate ATR(14)
    atr = np.full(n, np.nan)
    tr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(14, n):
        atr[i] = np.mean(tr[i-14:i])
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(14, n):
        highest_high[i] = np.max(high[i-14:i])
        lowest_low[i] = np.min(low[i-14:i])
    
    # Chop = log10(sum(atr(14))/abs(highest_high - lowest_low)) * log10(14) * 100
    chop = np.full(n, np.nan)
    for i in range(14, n):
        if highest_high[i] != lowest_low[i] and atr[i] > 0:
            sum_atr = np.sum(atr[i-14:i])
            chop[i] = np.log10(sum_atr / abs(highest_high[i] - lowest_low[i])) * np.log10(14) * 100
        else:
            chop[i] = 50.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Determine market regime: Chop > 61.8 = ranging (good for mean reversion)
        ranging_market = chop[i] > 61.8
        
        # Breakout conditions with volume confirmation
        breakout_up = close[i] > camarilla_h3_aligned[i]
        breakout_down = close[i] < camarilla_l3_aligned[i]
        
        # Entry conditions: breakout with volume confirmation in ranging market
        long_entry = breakout_up and (vol_ratio[i] > 2.0) and ranging_market
        short_entry = breakout_down and (vol_ratio[i] > 2.0) and ranging_market
        
        # Exit conditions: price returns to midpoint between H3 and L3
        midpoint = (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2
        long_exit = close[i] < midpoint
        short_exit = close[i] > midpoint
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_vol_chop_v4"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-12 21:34
