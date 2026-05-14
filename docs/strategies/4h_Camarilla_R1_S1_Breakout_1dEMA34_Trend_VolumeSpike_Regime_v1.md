# Strategy: 4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_Regime_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.255 | +4.6% | -18.2% | 70 | FAIL |
| ETHUSDT | 0.077 | +22.5% | -26.3% | 64 | PASS |
| SOLUSDT | 0.858 | +141.1% | -21.5% | 60 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.363 | +12.1% | -11.1% | 22 | PASS |
| SOLUSDT | 0.120 | +7.1% | -11.3% | 19 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_Regime_v1
Hypothesis: Camarilla R1/S1 breakouts on 4h with 1d EMA34 trend filter and volume spike confirmation, enhanced with 4h chop regime filter (BW < 50th percentile) to avoid whipsaws in ranging markets. Uses discrete sizing (0.25) to limit trades (~25/year) and avoid fee drag. The 1d EMA34 provides smooth trend alignment, reducing whipsaws. Volume spike (>2.0x 20-bar avg) confirms breakout momentum. Designed for BTC/ETH robustness in bull/bear regimes via trend-following structure with strict entry conditions and regime filter to improve bear market performance.
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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA34 on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of prior day)
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 20-bar average volume for confirmation on 4h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Bollinger Band Width for chop regime filter on 4h
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle
    # Use 50th percentile of BB width over 100 bars as regime threshold
    bb_width_ma100 = pd.Series(bb_width).rolling(window=100, min_periods=100).mean().values
    chop_filter = bb_width < bb_width_ma100  # True when in low volatility (trending) regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34, volume MA20, and BB calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma20[i]) or
            np.isnan(chop_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Volume confirmation: current volume > 2.0x 20-bar average (strict filter)
            volume_confirm = volume[i] > 2.0 * vol_ma20[i]
            
            # Long: price breaks above Camarilla R1 in uptrend with volume spike and trending regime
            # Short: price breaks below Camarilla S1 in downtrend with volume spike and trending regime
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema34_1d_aligned[i]) and volume_confirm and chop_filter[i]
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema34_1d_aligned[i]) and volume_confirm and chop_filter[i]
            
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
            # Exit when price moves back below 1d EMA34 (trend reversal)
            exit_signal = close[i] < ema34_1d_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above 1d EMA34 (trend reversal)
            exit_signal = close[i] > ema34_1d_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_Regime_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 18:58
