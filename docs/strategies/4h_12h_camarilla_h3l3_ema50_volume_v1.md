# Strategy: 4h_12h_camarilla_h3l3_ema50_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.177 | +26.5% | -7.2% | 372 | PASS |
| ETHUSDT | 0.167 | +26.5% | -10.1% | 358 | PASS |
| SOLUSDT | -0.001 | +19.0% | -14.9% | 299 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.715 | -4.1% | -6.0% | 146 | FAIL |
| ETHUSDT | 0.909 | +15.9% | -4.9% | 140 | PASS |

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
    
    # Hypothesis: 4h Camarilla H3/L3 breakout with 12h EMA50 trend filter and volume confirmation (>2.0x average)
    # Camarilla pivot levels provide high-probability reversal/continuation points from intraday structure
    # 12h EMA50 filters for intermediate trend alignment to avoid counter-trend whipsaws
    # Volume spike >2.0x 20-period average confirms institutional participation
    # Exits on H3/L3 retest or trend reversal
    # Target: 20-30 trades/year (80-120 total over 4 years) for low fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate previous 4h bar's Camarilla levels (H3, L3)
    # Based on previous 4h bar's range
    camarilla_h3 = np.full(len(high_4h), np.nan)
    camarilla_l3 = np.full(len(low_4h), np.nan)
    
    for i in range(1, len(high_4h)):
        # Use previous bar's high/low/close for Camarilla calculation
        ph = high_4h[i-1]
        pl = low_4h[i-1]
        pc = close_4h[i-1]
        rang = ph - pl
        
        camarilla_h3[i] = pc + rang * 1.1 / 4  # H3 level
        camarilla_l3[i] = pc - rang * 1.1 / 4  # L3 level
    
    # Get 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 4h volume for confirmation (>2.0x 20-period average)
    vol_ma_4h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_4h[i] = np.mean(volume[i-20:i])
    volume_spike_4h = volume > (2.0 * vol_ma_4h)
    
    # Align all indicators to LTF (4h)
    h3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    l3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_4h_aligned[i]) or np.isnan(l3_4h_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_spike_4h[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > h3_4h_aligned[i]
        short_breakout = close[i] < l3_4h_aligned[i]
        
        # 12h trend filter (EMA50)
        bullish_trend = close[i] > ema50_12h_aligned[i]
        bearish_trend = close[i] < ema50_12h_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike_4h[i]
        short_entry = short_breakout and bearish_trend and volume_spike_4h[i]
        
        # Exit logic: price retests H3/L3 or trend reversal
        long_exit = (close[i] <= h3_4h_aligned[i] * 1.001) or not bullish_trend  # Retest H3 or trend change
        short_exit = (close[i] >= l3_4h_aligned[i] * 0.999) or not bearish_trend  # Retest L3 or trend change
        
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

name = "4h_12h_camarilla_h3l3_ema50_volume_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-13 00:15
