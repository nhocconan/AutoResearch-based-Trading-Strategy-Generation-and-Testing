# Strategy: 4h_1d_1w_Williams_Alligator_Signal

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.500 | -37.0% | -38.9% | 368 | FAIL |
| ETHUSDT | 0.066 | +22.0% | -29.3% | 145 | PASS |
| SOLUSDT | 0.196 | +31.8% | -34.7% | 441 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.803 | +24.2% | -14.1% | 28 | PASS |
| SOLUSDT | 0.157 | +7.6% | -12.6% | 325 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_1d_1w_Williams_Alligator_Signal
Hypothesis: Williams Alligator (3 SMAs) on 1d defines trend, 1w filters volatility regime, 4h for entry timing.
In low volatility (1w ATR < 50th percentile): wait for Alligator alignment + 4h pullback to middle SMA.
In high volatility: trade Alligator breakouts on 4h with volume confirmation.
Designed to work in both bull and bear by adapting to market conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_Williams_Alligator_Signal"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY DATA FOR REGIME ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly ATR(14) for regime detection
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Weekly ATR percentile (50-period lookback) - regime filter
    atr_series = pd.Series(atr_1w)
    atr_percentile = atr_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    atr_regime = align_htf_to_ltf(prices, df_1w, atr_percentile)  # < 0.5 = low vol regime
    
    # === DAILY DATA FOR WILLIAMS ALLIGATOR ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs of median price
    median_price_1d = (high_1d + low_1d) / 2
    jaw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values  # Blue line
    teeth = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values   # Red line
    lips = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values    # Green line
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Alligator alignment: bullish when lips > teeth > jaw, bearish when lips < teeth < jaw
    bullish_aligned = lips_aligned > teeth_aligned
    bearish_aligned = lips_aligned < teeth_aligned
    
    # === 4H DATA FOR ENTRY TIMING ===
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Need enough lookback for indicators
        # Skip if not ready
        if (np.isnan(atr_regime[i]) or np.isnan(bullish_aligned[i]) or 
            np.isnan(bearish_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime: low volatility (< 50th percentile) = wait for alignment + pullback
        low_vol_regime = atr_regime[i] < 0.5
        
        if low_vol_regime:
            # LOW VOL: Wait for Alligator alignment, then enter on pullback to teeth
            bullish_setup = bullish_aligned[i] and close[i] <= teeth_aligned[i] and close[i] >= jaw_aligned[i]
            bearish_setup = bearish_aligned[i] and close[i] >= teeth_aligned[i] and close[i] <= jaw_aligned[i]
            
            # Exit when Alligator starts to sleep (lines converge)
            sleep_condition = (np.abs(lips_aligned[i] - teeth_aligned[i]) < 0.001 * close[i]) and \
                              (np.abs(teeth_aligned[i] - jaw_aligned[i]) < 0.001 * close[i])
            exit_long = sleep_condition
            exit_short = sleep_condition
            
        else:
            # HIGH VOL: Trade Alligator breakouts with volume confirmation
            bullish_breakout = close[i] > lips_aligned[i] and vol_ratio[i] > 1.5
            bearish_breakout = close[i] < jaw_aligned[i] and vol_ratio[i] > 1.5
            
            bullish_setup = bullish_breakout
            bearish_setup = bearish_breakout
            
            # Exit when price returns to Alligator mouth (between teeth and jaw)
            exit_long = close[i] <= teeth_aligned[i] or close[i] >= jaw_aligned[i]
            exit_short = close[i] >= teeth_aligned[i] or close[i] <= jaw_aligned[i]
        
        # Execute trades
        if bullish_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
```

## Last Updated
2026-04-12 06:31
