# Strategy: 6h_1d_camarilla_breakout_trend_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.283 | +37.1% | -12.9% | 65 | PASS |
| ETHUSDT | 0.042 | +18.7% | -17.9% | 72 | PASS |
| SOLUSDT | 0.967 | +199.5% | -30.4% | 58 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.060 | +4.6% | -6.3% | 25 | FAIL |
| ETHUSDT | 1.045 | +30.7% | -8.4% | 19 | PASS |
| SOLUSDT | 0.182 | +8.4% | -15.6% | 22 | PASS |

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
    
    # Hypothesis: 6h Camarilla pivot breakout with 1d trend filter + volume confirmation
    # Uses 1d Camarilla levels (calculated from prior 1d bar) for breakout/continuation
    # 1d EMA50 trend filter: only take breakouts in direction of daily trend
    # Volume confirmation: volume > 2.0 * 20-period average to filter false breakouts
    # Discrete sizing 0.25 to minimize fee churn. Target: 15-25 trades/year per symbol.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1d Camarilla levels (based on prior 1d bar's range)
    # Camarilla: H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    # H2 = close + 1.1*(high-low)/6, L2 = close - 1.1*(high-low)/6
    # H1 = close + 1.1*(high-low)/12, L1 = close - 1.1*(high-low)/12
    camarilla_h4 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        daily_range = high_1d[i-1] - low_1d[i-1]
        if daily_range > 0:
            camarilla_h4[i] = close_1d[i-1] + 1.1 * daily_range / 2
            camarilla_l4[i] = close_1d[i-1] - 1.1 * daily_range / 2
            camarilla_h3[i] = close_1d[i-1] + 1.1 * daily_range / 4
            camarilla_l3[i] = close_1d[i-1] - 1.1 * daily_range / 4
    
    # Align Camarilla levels to 6h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1d trend
        bullish_trend = close[i] > ema50_1d_aligned[i]
        bearish_trend = close[i] < ema50_1d_aligned[i]
        
        # Entry logic: Camarilla breakout with volume and trend filter
        long_entry = False
        short_entry = False
        
        # Long breakout: price breaks above H3/H4 in bullish trend
        if bullish_trend:
            long_entry = ((close[i] > h3_aligned[i-1]) or (close[i] > h4_aligned[i-1])) and volume_spike[i]
        # Short breakout: price breaks below L3/L4 in bearish trend
        elif bearish_trend:
            short_entry = ((close[i] < l3_aligned[i-1]) or (close[i] < l4_aligned[i-1])) and volume_spike[i]
        
        # Exit logic: opposite Camarilla level or trend reversal
        long_exit = (bearish_trend and close[i] < l3_aligned[i]) or (not bullish_trend and not bearish_trend)
        short_exit = (bullish_trend and close[i] > h3_aligned[i]) or (not bullish_trend and not bearish_trend)
        
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

name = "6h_1d_camarilla_breakout_trend_volume_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-12 21:48
