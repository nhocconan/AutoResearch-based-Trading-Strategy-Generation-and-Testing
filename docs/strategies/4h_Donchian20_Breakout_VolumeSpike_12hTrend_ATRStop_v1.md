# Strategy: 4h_Donchian20_Breakout_VolumeSpike_12hTrend_ATRStop_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.168 | +28.8% | -13.3% | 99 | PASS |
| ETHUSDT | 0.185 | +30.7% | -14.2% | 95 | PASS |
| SOLUSDT | 1.271 | +312.0% | -27.1% | 86 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.755 | -4.0% | -9.5% | 40 | FAIL |
| ETHUSDT | 0.591 | +17.8% | -7.9% | 31 | PASS |
| SOLUSDT | -0.053 | +3.1% | -16.8% | 30 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_12hTrend_ATRStop_v1
Hypothesis: On 4h timeframe, trade Donchian(20) breakouts with 12h EMA50 trend filter and volume spike confirmation. ATR-based stoploss limits drawdown. Target 20-50 trades/year by requiring confluence of HTF trend, volume confirmation, and price structure breakout. Designed to work in both bull and bear markets via trend filter and avoiding low-volume false breakouts.
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
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) from 4h data (HLC of 20 periods ago)
    donch_high_20 = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    donch_low_20 = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Get 12h data for HTF trend (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # ATR(14) for stoploss
    tr1 = pd.Series(high).rolling(window=2).max() - pd.Series(low).rolling(window=2).min()
    tr2 = abs(pd.Series(high).rolling(window=2).max() - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low).rolling(window=2).min() - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    donch_high_20_aligned = align_htf_to_ltf(prices, df_4h, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_4h, donch_low_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    # Warmup: max of EMA(50) 12h, Donchian(20) (need 21 bars for shift+window), volume MA (20), ATR(14)
    start_idx = max(50, 21, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(donch_high_20_aligned[i]) or
            np.isnan(donch_low_20_aligned[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_12h_val = ema_50_12h_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike = volume_spike[i]
        donch_high = donch_high_20_aligned[i]
        donch_low = donch_low_20_aligned[i]
        atr_val = atr[i]
        
        # Trend filter: price > EMA50 (uptrend) or < EMA50 (downtrend)
        uptrend = close_val > ema_50_12h_val
        downtrend = close_val < ema_50_12h_val
        
        if position == 0:
            # Long: break above Donchian high with uptrend and volume spike
            long_signal = (high_val > donch_high) and \
                          uptrend and \
                          vol_spike
            
            # Short: break below Donchian low with downtrend and volume spike
            short_signal = (low_val < donch_low) and \
                           downtrend and \
                           vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                atr_at_entry = atr_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                atr_at_entry = atr_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: ATR-based stoploss or trend reversal
            if close_val < entry_price - 2.0 * atr_at_entry or close_val < ema_50_12h_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: ATR-based stoploss or trend reversal
            if close_val > entry_price + 2.0 * atr_at_entry or close_val > ema_50_12h_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_12hTrend_ATRStop_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-26 03:29
