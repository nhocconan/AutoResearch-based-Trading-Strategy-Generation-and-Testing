# Strategy: 4h_Camarilla_H3L3_1dEMA34_Trend_VolumeConfirmation_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.343 | +40.5% | -12.5% | 239 | PASS |
| ETHUSDT | 0.079 | +22.1% | -17.5% | 235 | PASS |
| SOLUSDT | 0.835 | +144.6% | -30.9% | 236 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.687 | -2.2% | -9.7% | 84 | FAIL |
| ETHUSDT | 0.622 | +17.5% | -9.6% | 80 | PASS |
| SOLUSDT | 0.093 | +6.5% | -15.9% | 79 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 4h to target 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Camarilla levels: H3 = close + 1.1*(high-low)*1.1/12, L3 = close - 1.1*(high-low)*1.1/12 (from prior 1d).
- Entry: Long when price breaks above Camarilla H3 AND 1d EMA34 bullish AND volume > 1.3 * volume MA(20).
         Short when price breaks below Camarilla L3 AND 1d EMA34 bearish AND volume > 1.3 * volume MA(20).
- Exit: ATR-based trailing stop - exit long when price < highest_high_since_entry - 2.5*ATR,
        exit short when price > lowest_low_since_entry + 2.5*ATR.
- Signal size: 0.25 discrete to balance return and drawdown.
This strategy captures institutional-level breakouts in the direction of the daily trend with volume confirmation,
using wider ATR trailing stops to let winners run while controlling risk. Works in both bull and bear markets
by only taking trades in the direction of the 1d trend.
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
    
    # Get 1d data for EMA34 trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate prior day's Camarilla H3 and L3 levels
    # H3 = prior_day_close + 1.1*(prior_day_high - prior_day_low)*1.1/12
    # L3 = prior_day_close - 1.1*(prior_day_high - prior_day_low)*1.1/12
    prior_close = df_1d['close'].shift(1).values
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    camarilla_h3 = prior_close + 1.1 * (prior_high - prior_low) * 1.1 / 12
    camarilla_l3 = prior_close - 1.1 * (prior_high - prior_low) * 1.1 / 12
    
    # Calculate ATR(20) for trailing stop
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 34, 20)  # Need enough bars for EMA34, ATR/Vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation
            vol_confirmed = curr_volume > 1.3 * vol_ma[i]
            
            # Long: Price breaks above Camarilla H3 AND 1d EMA34 bullish AND volume confirmed
            if curr_close > camarilla_h3_aligned[i] and curr_close > ema_1d_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            # Short: Price breaks below Camarilla L3 AND 1d EMA34 bearish AND volume confirmed
            elif curr_close < camarilla_l3_aligned[i] and curr_close < ema_1d_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
        elif position == 1:
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR trailing stop: exit when price < highest_high - 2.5*ATR
            if curr_close < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: exit when price > lowest_low + 2.5*ATR
            if curr_close > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_1dEMA34_Trend_VolumeConfirmation_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-24 14:14
