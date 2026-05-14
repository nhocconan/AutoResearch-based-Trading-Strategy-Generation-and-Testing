# Strategy: 4h_Camarilla_R1_S1_Breakout_1dATR_Trend_VolumeSpike_v4

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.096 | +14.7% | -12.8% | 365 | FAIL |
| ETHUSDT | 0.293 | +37.0% | -10.3% | 321 | PASS |
| SOLUSDT | 0.372 | +51.8% | -25.0% | 273 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.003 | +23.1% | -9.2% | 118 | PASS |
| SOLUSDT | -0.005 | +5.0% | -15.4% | 101 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dATR_Trend_VolumeSpike_v4
Hypothesis: Use 1d ATR-based trend filter (price vs ATR-weighted close) with Camarilla R1/S1 breakouts to adapt to volatility regimes. Volume confirmation ensures institutional participation. Designed for low trade frequency (<50/year) to minimize fee drag while maintaining edge in both bull/bear regimes via volatility-adaptive trend filter.
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
    
    # Get 1d data for ATR trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d ATR (14-period) for volatility measurement and trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1d ATR-weighted trend filter: close > (ema_close + 0.5*atr) = uptrend
    ema_close_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_filter_1d = ema_close_1d + 0.5 * atr_14_1d  # Uptrend threshold
    
    # Previous 1d bar's high, low, close for Camarilla levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels: R1, S1
    camarilla_range = prev_high - prev_low
    R1 = prev_close + camarilla_range * 1.0/12
    S1 = prev_close - camarilla_range * 1.0/12
    
    # Align 1d indicators to 4h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    trend_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_filter_1d)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: 2.0x average volume (tighter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop (14-period on 4h)
    tr1_4h = high[1:] - low[1:]
    tr2_4h = np.abs(high[1:] - close[:-1])
    tr3_4h = np.abs(low[1:] - close[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    atr_14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Warmup: max of 1d ATR (14), trend filter (20), volume MA (20), 4h ATR (14)
    start_idx = max(14, 20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(trend_filter_1d_aligned[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr_14_4h[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        atr_14_1d_val = atr_14_1d_aligned[i]
        trend_filter_1d_val = trend_filter_1d_aligned[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_14_4h_val = atr_14_4h[i]
        
        if position == 0:
            # Long: break above R1, uptrend (close > trend_filter), volume spike
            long_signal = (high_val > R1_val) and (close_val > trend_filter_1d_val) and (volume_val > 2.0 * vol_ma_val)
            # Short: break below S1, downtrend (close < trend_filter), volume spike
            short_signal = (low_val < S1_val) and (close_val < trend_filter_1d_val) and (volume_val > 2.0 * vol_ma_val)
            
            if long_signal:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.0 * atr_14_4h_val
            elif short_signal:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.0 * atr_14_4h_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.0 * atr_14_4h_val)
            # Exit: trailing stop hit or trend reversal (close < trend_filter)
            if (low_val < long_stop) or (close_val < trend_filter_1d_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.0 * atr_14_4h_val)
            # Exit: trailing stop hit or trend reversal (close > trend_filter)
            if (high_val > short_stop) or (close_val > trend_filter_1d_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dATR_Trend_VolumeSpike_v4"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-26 01:32
