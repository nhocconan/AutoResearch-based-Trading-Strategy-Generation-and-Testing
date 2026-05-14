# Strategy: 4h_Camarilla_R1S1_Breakout_1dTrend_VolumeChop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.434 | +45.0% | -15.8% | 410 | PASS |
| ETHUSDT | 0.155 | +28.0% | -15.8% | 380 | PASS |
| SOLUSDT | 1.112 | +199.6% | -27.1% | 317 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.327 | +1.9% | -8.4% | 130 | FAIL |
| ETHUSDT | 0.860 | +22.7% | -11.5% | 130 | PASS |
| SOLUSDT | 0.937 | +25.0% | -11.3% | 113 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeChop
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA50 trend filter and chop regime filter.
Long when price breaks above R1 in non-choppy uptrend (close > 1d EMA50 and CHOP < 61.8).
Short when price breaks below S1 in non-choppy downtrend (close < 1d EMA50 and CHOP < 61.8).
Volume confirmation: current volume > 1.5x 20-bar average.
Exit when price re-enters H3-L3 range or chop increases (CHOP > 61.8) indicating ranging market.
Uses discrete position sizing (0.30) to minimize fee churn and target ~20-40 trades/year.
Designed to work in trending markets (bull/bear) and avoid whipsaws in ranging markets via chop filter.
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
    
    # Volume regime: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (1.5 * vol_ma_20)
    
    # Choppiness Index regime filter (using 14-period)
    # CHOP > 61.8 = ranging/choppy market (avoid), CHOP < 38.2 = trending
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    # Avoid division by zero
    chop_denominator = highest_high - lowest_low
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)
    chop = 100 * np.log10(atr * np.sqrt(atr_period) / chop_denominator) / np.log10(atr_period)
    chop_regime = chop < 61.8  # True when not choppy (trending market)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(100, atr_period)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        
        if position == 0:
            # Only trade in non-choppy (trending) regimes
            if chop_regime[i]:
                if close[i] > ema_trend:  # Uptrend regime (1d)
                    # Long: break above R1 with volume regime
                    long_signal = (close[i] > r1_aligned[i]) and vol_regime[i]
                    # Short: break below S1 only if extreme volume (counter-trend fade on high volume)
                    short_signal = (close[i] < s1_aligned[i]) and (volume[i] > (3.0 * vol_ma_20[i]))
                else:  # Downtrend regime (1d)
                    # Short: break below S1 with volume regime
                    short_signal = (close[i] < s1_aligned[i]) and vol_regime[i]
                    # Long: break above R1 only if extreme volume (counter-trend fade on high volume)
                    long_signal = (close[i] > r1_aligned[i]) and (volume[i] > (3.0 * vol_ma_20[i]))
            else:
                long_signal = False
                short_signal = False
            
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
            # Exit conditions: re-enter H3-L3 range or chop increases (ranging market)
            exit_signal = (close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or (chop[i] > 61.8)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit conditions: re-enter H3-L3 range or chop increases (ranging market)
            exit_signal = (close[i] > l3_aligned[i] and close[i] < h3_aligned[i]) or (chop[i] > 61.8)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeChop"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 21:15
