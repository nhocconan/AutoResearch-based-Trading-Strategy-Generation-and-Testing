# Strategy: 4h_Camarilla_R1S1_Breakout_1dTrend_VolumeFilter_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.422 | +13.0% | -4.9% | 141 | FAIL |
| ETHUSDT | 0.321 | +30.6% | -3.7% | 109 | PASS |
| SOLUSDT | 0.326 | +35.4% | -13.2% | 102 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.097 | +6.8% | -5.4% | 47 | PASS |
| SOLUSDT | 0.034 | +6.3% | -4.4% | 37 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeFilter_v2
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA200 trend filter and volume spike confirmation.
Uses tighter volume threshold (2.5x) and discrete sizing (0.25) to reduce trades to ~25/year.
In uptrend (close > 1d EMA200): long R1 break, short S1 fade only with extreme volume.
In downtrend: short S1 break, long R1 fade only with extreme volume.
Exit on opposite Camarilla level touch or trend reversal. Designed for low fee drag and robustness in bull/bear.
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
    
    # Get 4h data for Camarilla calculations (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar (based on previous bar)
    R1_4h = np.full(len(close_4h), np.nan)
    S1_4h = np.full(len(close_4h), np.nan)
    
    for i in range(1, len(close_4h)):
        # Camarilla levels based on previous 4h bar's range
        high_prev = high_4h[i-1]
        low_prev = low_4h[i-1]
        close_prev = close_4h[i-1]
        range_prev = high_prev - low_prev
        
        if range_prev > 0:
            R1_4h[i] = close_prev + (range_prev * 1.1 / 4)  # R1 level
            S1_4h[i] = close_prev - (range_prev * 1.1 / 4)  # S1 level
    
    # Align Camarilla levels to original timeframe
    R1_4h_aligned = align_htf_to_ltf(prices, df_4h, R1_4h)
    S1_4h_aligned = align_htf_to_ltf(prices, df_4h, S1_4h)
    
    # Get 1d data for trend filter (EMA200)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA200 for trend direction
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: volume > 2.5x 20-period average (stricter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_4h_aligned[i]) or np.isnan(S1_4h_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_200_1d_aligned[i]
        
        if position == 0:
            # Regime-based entry logic
            if close[i] > ema_trend:  # Uptrend regime
                # Long: break above R1 with volume spike
                long_signal = (close[i] > R1_4h_aligned[i]) and vol_spike[i]
                # Short: break below S1 only if extreme volume spike (counter-trend fade)
                short_signal = (close[i] < S1_4h_aligned[i]) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            else:  # Downtrend regime
                # Short: break below S1 with volume spike
                short_signal = (close[i] < S1_4h_aligned[i]) and vol_spike[i]
                # Long: break above R1 only if extreme volume spike (counter-trend fade)
                long_signal = (close[i] > R1_4h_aligned[i]) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            
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
            # Exit conditions: touch S1 or trend reversal
            exit_signal = (close[i] < S1_4h_aligned[i]) or (close[i] < ema_trend * 0.995)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: touch R1 or trend reversal
            exit_signal = (close[i] > R1_4h_aligned[i]) or (close[i] > ema_trend * 1.005)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeFilter_v2"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 20:53
