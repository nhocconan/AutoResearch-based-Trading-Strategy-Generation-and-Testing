# Strategy: 4h_Camarilla_H3L3_Breakout_12hEMA50_Trend_VolumeSpike_ChopFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.216 | +30.3% | -10.0% | 199 | PASS |
| ETHUSDT | 0.133 | +26.5% | -14.0% | 194 | PASS |
| SOLUSDT | 0.764 | +105.3% | -24.7% | 162 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.009 | -4.4% | -7.4% | 75 | FAIL |
| ETHUSDT | 0.786 | +19.1% | -12.3% | 67 | PASS |
| SOLUSDT | 0.030 | +5.8% | -10.6% | 55 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout + 12h EMA50 Trend + Volume Spike + Chop Filter
Hypothesis: Camarilla H3/L3 levels provide strong intraday support/resistance.
Breaking above H3 with volume spike in uptrend (price > EMA50) captures momentum.
Breaking below L3 with volume spike in downtrend (price < EMA50) captures short opportunities.
Choppiness filter avoids whipsaws in ranging markets. Discrete sizing (0.0, ±0.25) minimizes fee churn.
Target: 30-60 trades/year on 4h timeframe. Works in both bull and bear via trend alignment.
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
    
    # Get 12h data for EMA50 and 1d data for Camarilla (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close for trend
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels on 1d (based on previous day's high/low/close)
    # H3 = Close + 1.1 * (High - Low) / 4
    # L3 = Close - 1.1 * (High - Low) / 4
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    # Align Camarilla levels to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate Choppiness Index (14) to avoid ranging markets
    # CHOP = 100 * log10(sum(ATR14) / (HHV14 - LLV14)) / log10(14)
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14 * 14 / (hh14 - ll14 + 1e-10)) / np.log10(14)
    chop_filter = chop < 61.8  # only allow trades when not strongly ranging (CHOP < 61.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA, ATR, and CHOP
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        h3 = h3_aligned[i]
        l3 = l3_aligned[i]
        ema_trend = ema_50_12h_aligned[i]
        vol_spike = volume_spike[i]
        not_choppy = chop_filter[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 AND volume spike AND price > EMA (uptrend) AND not choppy
            long_entry = (curr_close > h3) and vol_spike and (curr_close > ema_trend) and not_choppy
            # Short: price breaks below L3 AND volume spike AND price < EMA (downtrend) AND not choppy
            short_entry = (curr_close < l3) and vol_spike and (curr_close < ema_trend) and not_choppy
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below L3 OR price crosses below EMA
            if (curr_close < l3) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above H3 OR price crosses above EMA
            if (curr_close > h3) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_12hEMA50_Trend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 06:05
