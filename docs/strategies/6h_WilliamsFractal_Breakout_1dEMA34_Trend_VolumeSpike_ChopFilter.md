# Strategy: 6h_WilliamsFractal_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.343 | +35.8% | -11.6% | 40 | PASS |
| ETHUSDT | 0.507 | +48.5% | -11.3% | 35 | PASS |
| SOLUSDT | 1.171 | +180.4% | -16.1% | 40 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.960 | -0.5% | -5.7% | 17 | FAIL |
| ETHUSDT | 0.378 | +10.3% | -6.8% | 13 | PASS |
| SOLUSDT | -0.208 | +2.9% | -7.1% | 12 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h Williams Fractal Breakout + 1d EMA34 Trend + Volume Spike + Chop Regime Filter
Hypothesis: Williams Fractals identify swing points on 1d chart. Breakouts above recent bullish fractal or below bearish fractal on 6h, with 1d EMA34 trend alignment, volume confirmation, and non-choppy regime capture momentum moves. Works in bull/bear via trend filter and discrete sizing (0.25). Targets 50-150 trades over 4 years on 6h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for fractals, EMA34, and Chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1d Chop regime filter: CHOP(14) > 61.8 = range (avoid), < 38.2 = trend (favor)
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        tr1 = np.abs(high_arr[1:] - low_arr[1:])
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        
        highest_high = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        hhll = highest_high - lowest_low
        
        atr_sum = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        chop = 100 * np.log10(atr_sum / np.log(10) / hhll)
        return chop
    
    chop_values = calculate_chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Williams Fractals on 1d (requires 2 extra bars for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align with 2-bar delay for fractal confirmation (needs 2 future 1d bars to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 1d EMA warmup, fractals, Chop, and volume MA
    start_idx = max(50, 21)  # EMA34 needs ~34, plus buffers
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Regime filter: only trade when NOT choppy (CHOP < 61.8 = trending)
        not_choppy = chop_aligned[i] < 61.8
        
        # Trend filter: price relative to 1d EMA34
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Fractal breakout + trend + volume + regime
            # Long: price breaks above recent bullish fractal AND bullish bias AND volume spike AND not choppy
            long_entry = (curr_high > bullish_fractal_aligned[i]) and bullish_bias and vol_spike and not_choppy
            # Short: price breaks below recent bearish fractal AND bearish bias AND volume spike AND not choppy
            short_entry = (curr_low < bearish_fractal_aligned[i]) and bearish_bias and vol_spike and not_choppy
            
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
            # Exit: price falls below recent bearish fractal (invalidates uptrend) OR loss of bullish bias OR choppy regime
            if (curr_low < bearish_fractal_aligned[i]) or (curr_close < ema_1d_aligned[i]) or (chop_aligned[i] >= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above recent bullish fractal (invalidates downtrend) OR loss of bearish bias OR choppy regime
            if (curr_high > bullish_fractal_aligned[i]) or (curr_close > ema_1d_aligned[i]) or (chop_aligned[i] >= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 07:44
