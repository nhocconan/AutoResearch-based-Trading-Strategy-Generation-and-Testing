# Strategy: 6h_ElderRay_1dRegime_Filter_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.279 | +3.4% | -17.6% | 541 | FAIL |
| ETHUSDT | 0.137 | +26.7% | -21.5% | 530 | PASS |
| SOLUSDT | 0.995 | +192.5% | -32.9% | 535 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.587 | +17.2% | -8.6% | 163 | PASS |
| SOLUSDT | -0.441 | -5.3% | -17.2% | 175 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h_ElderRay_1dRegime_Filter_v2
Hypothesis: Trade 6h Elder Ray (Bull/Bear Power) with 1d trend regime filter and volume confirmation.
Elder Ray measures bull/bear power relative to EMA13. Long when Bull Power > 0 and rising, 
Short when Bear Power < 0 and falling. Use 1d EMA34 for trend regime (bull/bear/range).
Only trade in direction of 1d trend: long in bull regime, short in bear regime, flat in range.
Add volume confirmation (volume > 1.5 * ATR) to avoid false signals.
Target: 12-30 trades/year to minimize fee drag while capturing sustained moves.
Discrete sizing: 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend regime and EMA13 for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend regime
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate EMA13 for Elder Ray (need 1d high/low/close)
    ema_13_1d_high = pd.Series(df_1d['high'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_low = pd.Series(df_1d['low'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_close = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align 1d EMA13 to 6h timeframe
    ema_13_1d_high_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d_high)
    ema_13_1d_low_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d_low)
    ema_13_1d_close_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d_close)
    
    # Calculate ATR for volume spike filter (using 6h data)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # first TR undefined
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # track holding period
    
    # Start index: need warmup for 1d EMA34 (34) and EMA13 (13) and ATR (14)
    start_idx = max(34, 13, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_13_1d_high_aligned[i]) or np.isnan(ema_13_1d_low_aligned[i]) or np.isnan(ema_13_1d_close_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Calculate Elder Ray components
        bull_power = high[i] - ema_13_1d_close_aligned[i]  # Bull Power = High - EMA13(close)
        bear_power = low[i] - ema_13_1d_close_aligned[i]   # Bear Power = Low - EMA13(close)
        
        # Volume spike: current volume > 1.5 * ATR (adaptive threshold)
        volume_spike = volume[i] > 1.5 * atr[i]
        
        # Determine 1d trend regime
        # Bull regime: price > EMA34
        # Bear regime: price < EMA34
        # Range regime: near EMA34 (within 0.5*ATR of 6h equivalent)
        # Convert 1d EMA34 to 6h equivalent threshold: use 1d ATR scaled to 6h
        # Approximate: 1d ATR ≈ 6h ATR * sqrt(4) since 1d = 4*6h bars
        atr_6h = atr[i]
        atr_1d_approx = atr_6h * 2.0  # rough approximation
        regime_threshold = 0.5 * atr_1d_approx
        
        if close[i] > ema_34_1d_aligned[i] + regime_threshold:
            regime = 'bull'  # only allow longs
        elif close[i] < ema_34_1d_aligned[i] - regime_threshold:
            regime = 'bear'  # only allow shorts
        else:
            regime = 'range'  # no trades
        
        if position == 0:
            # Long setup: Bull Power > 0 and rising (vs previous bar) AND volume spike AND bull regime
            if i > start_idx:
                bull_power_prev = high[i-1] - ema_13_1d_close_aligned[i-1]
                bull_power_rising = bull_power > bull_power_prev
            else:
                bull_power_rising = False
            
            long_setup = (bull_power > 0) and bull_power_rising and volume_spike and (regime == 'bull')
            
            # Short setup: Bear Power < 0 and falling (vs previous bar) AND volume spike AND bear regime
            if i > start_idx:
                bear_power_prev = low[i-1] - ema_13_1d_close_aligned[i-1]
                bear_power_falling = bear_power < bear_power_prev
            else:
                bear_power_falling = False
            
            short_setup = (bear_power < 0) and bear_power_falling and volume_spike and (regime == 'bear')
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            elif short_setup:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            bars_since_entry += 1
            # Exit: Bull Power <= 0 OR regime turns bearish OR min holding period (8 bars = 1 day)
            if (bull_power <= 0) or (regime == 'bear') or (bars_since_entry >= 8):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            bars_since_entry += 1
            # Exit: Bear Power >= 0 OR regime turns bullish OR min holding period (8 bars = 1 day)
            if (bear_power >= 0) or (regime == 'bull') or (bars_since_entry >= 8):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
    
    return signals

name = "6h_ElderRay_1dRegime_Filter_v2"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 14:15
