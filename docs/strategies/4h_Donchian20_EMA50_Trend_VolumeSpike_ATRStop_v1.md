# Strategy: 4h_Donchian20_EMA50_Trend_VolumeSpike_ATRStop_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.189 | +30.4% | -16.2% | 130 | PASS |
| ETHUSDT | 0.001 | +15.4% | -18.2% | 123 | PASS |
| SOLUSDT | 1.225 | +298.4% | -25.8% | 126 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.137 | -8.1% | -13.3% | 53 | FAIL |
| ETHUSDT | 0.778 | +22.6% | -7.8% | 37 | PASS |
| SOLUSDT | 0.543 | +17.7% | -10.9% | 37 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h Donchian(20) breakout + 12h EMA50 trend + volume spike + ATR stoploss
Hypothesis: Donchian breakouts capture strong momentum. 12h EMA50 filters trend direction.
Volume spike confirms institutional participation. ATR-based stoploss manages risk.
Works in bull/bear via trend filter and volatility-based position sizing.
Target: 20-50 trades/year on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(14) for stoploss and volatility filter
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Calculate Donchian channels (20-period)
    if len(close) >= 20:
        donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
        donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    else:
        donch_high = np.full(n, np.nan)
        donch_low = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50 = ema_50_aligned[i]
        upper = donch_high[i]
        lower = donch_low[i]
        atr_val = atr[i]
        
        # Volume spike: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 1.5 * vol_ma_20
        
        # Trend filter
        uptrend = curr_close > ema_50
        downtrend = curr_close < ema_50
        
        if position == 0:
            # Long: price breaks above upper Donchian AND volume spike AND uptrend
            long_condition = (curr_high > upper) and volume_spike and uptrend
            # Short: price breaks below lower Donchian AND volume spike AND downtrend
            short_condition = (curr_low < lower) and volume_spike and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2*ATR below entry) or trend reversal
            if curr_close <= entry_price - 2.0 * atr_val or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2*ATR above entry) or trend reversal
            if curr_close >= entry_price + 2.0 * atr_val or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_EMA50_Trend_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 01:40
