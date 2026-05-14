# Strategy: 4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeRegime

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.398 | +42.0% | -11.2% | 420 | PASS |
| ETHUSDT | 0.433 | +48.4% | -13.1% | 385 | PASS |
| SOLUSDT | 1.061 | +184.2% | -27.1% | 326 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.125 | +7.2% | -5.3% | 135 | PASS |
| ETHUSDT | 0.914 | +23.6% | -10.2% | 132 | PASS |
| SOLUSDT | 1.225 | +31.3% | -9.8% | 115 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeRegime
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume regime filter.
Long when price breaks above R1 in uptrend (close > 1d EMA50) with elevated volume (>1.5x 20-bar average).
Short when price breaks below S1 in downtrend (close < 1d EMA50) with elevated volume.
Exit when price re-enters H3-L3 range or trend reverses.
Uses discrete position sizing (0.30) to minimize fee churn and keep trades ~20-40/year.
Designed to work in bull markets via trend-following breakouts and in bear markets via counter-trend fades on extreme volume spikes.
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
    
    # Get 1d data for Camarilla pivot calculation and trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous 1d bar
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    r1 = prev_close + range_hl * 1.1 / 12
    s1 = prev_close - range_hl * 1.1 / 12
    h3 = prev_close + range_hl * 1.1 / 4
    l3 = prev_close - range_hl * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Get 1d data for trend filter (EMA50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume regime: volume > 1.5x 20-period average (avoid churn, confirm momentum)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        
        if position == 0:
            # Regime-based entry logic
            if close[i] > ema_trend:  # Uptrend regime (1d)
                # Long: break above R1 with volume regime
                long_signal = (close[i] > r1_aligned[i]) and vol_regime[i]
                # Short: break below S1 only if extreme volume (counter-trend fade)
                short_signal = (close[i] < s1_aligned[i]) and (volume[i] > (3.0 * vol_ma_20[i]))
            else:  # Downtrend regime (1d)
                # Short: break below S1 with volume regime
                short_signal = (close[i] < s1_aligned[i]) and vol_regime[i]
                # Long: break above R1 only if extreme volume (counter-trend fade)
                long_signal = (close[i] > r1_aligned[i]) and (volume[i] > (3.0 * vol_ma_20[i]))
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit conditions: re-enter H3-L3 range or trend reversal
            exit_signal = (close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or (close[i] < ema_trend * 0.98)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit conditions: re-enter H3-L3 range or trend reversal
            exit_signal = (close[i] > l3_aligned[i] and close[i] < h3_aligned[i]) or (close[i] > ema_trend * 1.02)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeRegime"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 21:13
