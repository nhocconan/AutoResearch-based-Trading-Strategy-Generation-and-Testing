# Strategy: 4h_Camarilla_R1_S1_Breakout_1dEMA34_TRIX_Confluence

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.011 | +21.0% | -5.8% | 191 | FAIL |
| ETHUSDT | 0.418 | +36.1% | -6.0% | 172 | PASS |
| SOLUSDT | 0.580 | +54.5% | -14.0% | 146 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.032 | +18.1% | -7.7% | 69 | PASS |
| SOLUSDT | 0.400 | +10.0% | -6.4% | 52 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_TRIX_Confluence
Hypothesis: Camarilla R1/S1 breakouts on 4h with 1d EMA34 trend filter and TRIX momentum confirmation. 
Only trades when TRIX crosses zero in direction of trend, reducing false breakouts. 
Volume spike (>2x 20-bar avg) confirms participation. 
Designed for low trade frequency (<40/year) to minimize fee drag and work in both bull/bear markets via trend alignment and momentum filter.
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
    
    # Get 1d data for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA34 on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate TRIX on 1d close (15-period EMA of 15-period EMA of 15-period EMA, then ROC)
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = pd.Series(ema3).pct_change(periods=1).values * 100  # ROC of triple EMA
    
    # Calculate Camarilla levels on 1d data (based on previous bar's OHLC)
    camarilla_r1_1d = close_1d + ((high_1d - low_1d) * 1.1 / 12)
    camarilla_s1_1d = close_1d - ((high_1d - low_1d) * 1.1 / 12)
    camarilla_h4_1d = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    camarilla_l4_1d = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # Align HTF indicators to 4h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d, additional_delay_bars=1)
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix, additional_delay_bars=1)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d, additional_delay_bars=1)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d, additional_delay_bars=1)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d, additional_delay_bars=1)
    
    # Volume confirmation: 2.0x 20-bar average volume (strict filter)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34) and TRIX (15*3=45) and volume MA (20)
    start_idx = max(45, 20)  # TRIX needs ~45 bars, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or 
            np.isnan(trix_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for breakout signals with trend filter, TRIX confirmation, and volume spike
            # Long: price breaks above R1 in uptrend (close > EMA34) with TRIX > 0 and volume spike
            # Short: price breaks below S1 in downtrend (close < EMA34) with TRIX < 0 and volume spike
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema34_aligned[i]) and (trix_aligned[i] > 0) and volume_spike[i]
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema34_aligned[i]) and (trix_aligned[i] < 0) and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below Camarilla H4 (take profit at resistance)
            exit_signal = close[i] < camarilla_h4_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Camarilla L4 (take profit at support)
            exit_signal = close[i] > camarilla_l4_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_TRIX_Confluence"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 18:21
