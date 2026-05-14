# Strategy: 4h_Volume_Spike_Keltner_Breakout_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.161 | +28.3% | -12.5% | 159 | PASS |
| ETHUSDT | 0.255 | +36.2% | -15.7% | 103 | PASS |
| SOLUSDT | 0.423 | +63.5% | -41.4% | 98 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.357 | -9.5% | -12.9% | 34 | FAIL |
| ETHUSDT | 0.433 | +13.2% | -9.7% | 57 | PASS |
| SOLUSDT | -0.514 | -6.0% | -22.8% | 26 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_Volume_Spike_Keltner_Breakout_v1
Hypothesis: Uses Keltner Channel breakout with volume spike confirmation and 1-day ADX trend filter.
Works in bull markets via breakouts and bear markets via mean reversion at channel edges during low volatility.
Target: 20-40 trades/year to minimize fee drag while capturing strong moves.
"""

name = "4h_Volume_Spike_Keltner_Breakout_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d ADX for trend filter (trending >25, ranging <20) ---
    # Calculate True Range
    tr1 = pd.Series(df_1d['high']).subtract(df_1d['low']).abs()
    tr2 = pd.Series(df_1d['high']).subtract(df_1d['close'].shift(1)).abs()
    tr3 = pd.Series(df_1d['low']).subtract(df_1d['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    dm_plus = pd.Series(df_1d['high']).diff()
    dm_minus = pd.Series(df_1d['low']).diff().abs()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr_1d = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_plus_smooth = dm_plus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_minus_smooth = dm_minus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_1d
    di_minus = 100 * dm_minus_smooth / atr_1d
    
    # Calculate DX and ADX
    dx = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100
    adx_1d = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_1d_values = adx_1d.values
    
    # Align ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_values)
    
    # --- Keltner Channel (20-period EMA, 2*ATR) ---
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean()
    
    # True Range for Keltner
    tr_keltner = np.maximum(
        high - low,
        np.maximum(
            abs(high - np.roll(close, 1)),
            abs(low - np.roll(close, 1))
        )
    )
    tr_keltner[0] = high[0] - low[0]  # First value
    atr_keltner = pd.Series(tr_keltner).ewm(span=10, adjust=False, min_periods=10).mean()
    
    upper_keltner = ema_20 + (2 * atr_keltner.values)
    lower_keltner = ema_20 - (2 * atr_keltner.values)
    
    # --- Volume Spike Detection ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)  # Significant volume spike
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20[i]) or 
            np.isnan(upper_keltner[i]) or
            np.isnan(lower_keltner[i]) or
            np.isnan(adx_1d_aligned[i]) or
            np.isnan(vol_spike[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine market regime based on ADX
        adx = adx_1d_aligned[i]
        is_trending = adx > 25
        is_ranging = adx < 20
        
        # Breakout signals
        long_breakout = (high[i] > upper_keltner[i]) and vol_spike[i]
        short_breakout = (low[i] < lower_keltner[i]) and vol_spike[i]
        
        # Mean reversion signals (only in ranging markets)
        long_reversion = (close[i] < lower_keltner[i]) and is_ranging
        short_reversion = (close[i] > upper_keltner[i]) and is_ranging
        
        if position == 0:
            if is_trending:
                # In trending markets, only take breakout signals
                if long_breakout:
                    signals[i] = 0.25
                    position = 1
                elif short_breakout:
                    signals[i] = -0.25
                    position = -1
            else:
                # In ranging markets, take both breakout and mean reversion
                if long_breakout or long_reversion:
                    signals[i] = 0.25
                    position = 1
                elif short_breakout or short_reversion:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price touches opposite Keltner band or ADX drops (trend weakening)
                exit_signal = (low[i] < lower_keltner[i]) or (adx < 20)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price touches opposite Keltner band or ADX drops
                exit_signal = (high[i] > upper_keltner[i]) or (adx < 20)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-11 05:35
