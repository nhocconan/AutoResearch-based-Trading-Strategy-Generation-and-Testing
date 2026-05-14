# Strategy: 4h_Camarilla_H3L3_1dEMA34_VolumeSpike_ATRTrail

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.366 | +36.7% | -10.3% | 181 | PASS |
| ETHUSDT | 0.707 | +64.8% | -9.4% | 175 | PASS |
| SOLUSDT | 0.862 | +116.9% | -15.5% | 175 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.631 | +11.8% | -4.5% | 66 | PASS |
| ETHUSDT | 1.129 | +23.9% | -5.9% | 66 | PASS |
| SOLUSDT | 0.963 | +21.0% | -6.7% | 54 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d Camarilla pivot (H3/L3) breakout with 1d EMA34 trend filter and volume confirmation
- Uses 1d EMA34 slope for trend bias (long when rising, short when falling)
- Breakout triggers when price closes beyond 1d H3 (long) or L3 (short) with volume > 1.8x 20-period 4h MA
- Fixed position size 0.25 to limit fee churn and manage drawdown
- ATR-based trailing stop (2.0x ATR) to lock in profits and reduce losses
- Camarilla levels work well in ranging markets; EMA34 filter ensures we only trade with the daily trend
- Designed to work in bull markets (buying H3 breakouts in uptrends) and bear markets (selling L3 breakdowns in downtrends)
- Tight entry conditions target 75-200 trades over 4 years to avoid fee drag
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
    
    # Get 1d data for Camarilla pivots and EMA34 (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day's range)
    # H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    # Using previous day's high/low to avoid look-ahead
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = high_1d[0]  # first period
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    camarilla_h3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_l3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    
    # Calculate 1d EMA34 and its slope
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_slope = np.gradient(ema34_1d)  # slope of EMA34
    
    # Get 4h data for volume confirmation and ATR (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Volume average (20-period) on 4h
    volume_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) on 4h for stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to 4h timeframe (primary)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    ema34_slope_aligned = align_htf_to_ltf(prices, df_1d, ema34_slope)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema34_slope_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        h3_val = h3_aligned[i]
        l3_val = l3_aligned[i]
        ema_slope = ema34_slope_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend filter
            # Long: price closes above H3 + volume spike + EMA34 rising
            if price > h3_val and vol > 1.8 * vol_ma and ema_slope > 0:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.0 * atr_val
            # Short: price closes below L3 + volume spike + EMA34 falling
            elif price < l3_val and vol > 1.8 * vol_ma and ema_slope < 0:
                signals[i] = -0.25
                position = -1
                entry_price = price
                atr_stop = entry_price + 2.0 * atr_val
        
        elif position == 1:
            # Check stoploss
            if price <= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Trail stop: raise stop if price moves favorably
                atr_stop = max(atr_stop, price - 1.5 * atr_val)
        
        elif position == -1:
            # Check stoploss
            if price >= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Trail stop: lower stop if price moves favorably
                atr_stop = min(atr_stop, price + 1.5 * atr_val)
    
    return signals

name = "4h_Camarilla_H3L3_1dEMA34_VolumeSpike_ATRTrail"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-17 21:28
