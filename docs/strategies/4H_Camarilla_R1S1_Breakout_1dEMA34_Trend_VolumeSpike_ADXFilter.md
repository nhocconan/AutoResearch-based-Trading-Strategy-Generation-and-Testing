# Strategy: 4H_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_ADXFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.134 | +18.8% | -4.8% | 256 | FAIL |
| ETHUSDT | 0.043 | +22.6% | -3.7% | 240 | PASS |
| SOLUSDT | -0.809 | -5.5% | -15.1% | 217 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.664 | +10.6% | -3.9% | 89 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
Optimized for trade count: reduced volume filter to 2.0x and added ADX regime filter to avoid overtrading.
Target: 20-40 trades/year per symbol. Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # Camarilla calculation: based on previous day's range
    range_1d = high_1d - low_1d
    r1 = close_1d_arr + 0.125 * range_1d  # R1 level
    s1 = close_1d_arr - 0.125 * range_1d  # S1 level
    h4 = close_1d_arr + 1.125 * range_1d  # H4 level
    l4 = close_1d_arr - 1.125 * range_1d  # L4 level
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ADX (14) for regime filter - avoid ranging markets
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # need EMA34, volume MA20, ADX14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1d EMA34 = uptrend, close < 1d EMA34 = downtrend
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: 4h volume > 2.0x 20-period MA (tighter to reduce trades)
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        # Regime filter: ADX > 25 to avoid ranging markets
        regime_filter = adx[i] > 25
        
        # Camarilla breakout conditions
        breakout_up = close[i] > r1_aligned[i]  # Break above R1
        breakout_down = close[i] < s1_aligned[i]  # Break below S1
        return_to_h4 = close[i] < h4_aligned[i]  # Return below H4 (exit long)
        return_to_l4 = close[i] > l4_aligned[i]  # Return above L4 (exit short)
        opposite_extreme = (position == 1 and breakout_down) or \
                           (position == -1 and breakout_up)
        
        if position == 0:
            # Long: Break above R1 AND uptrend AND volume confirmation AND trending regime
            if breakout_up and trend_up and vol_filter and regime_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 AND downtrend AND volume confirmation AND trending regime
            elif breakout_down and trend_down and vol_filter and regime_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to H4/L4 or opposite extreme hit
            exit_signal = False
            if position == 1:
                exit_signal = return_to_h4 or opposite_extreme
            elif position == -1:
                exit_signal = return_to_l4 or opposite_extreme
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_ADXFilter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 14:35
