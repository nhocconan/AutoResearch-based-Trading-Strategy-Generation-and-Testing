# Strategy: 4h_Donchian20_Breakout_1dATR_Regime_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.293 | +7.9% | -13.8% | 19 | FAIL |
| ETHUSDT | 0.068 | +22.6% | -17.5% | 20 | PASS |
| SOLUSDT | 1.244 | +219.4% | -18.7% | 16 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.403 | +10.9% | -7.0% | 8 | PASS |
| SOLUSDT | -1.098 | -6.9% | -15.5% | 7 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# Donchian channel breakouts capture strong momentum moves. ATR(1d) > ATR(20d) indicates
# high volatility regime favorable for breakouts. Volume confirmation ensures institutional
# participation. Works in bull (breakouts with volume) and bear (volatility expansion after
# consolidation). Discrete sizing (0.25) minimizes fee churn. Target: 75-200 total trades.

name = "4h_Donchian20_Breakout_1dATR_Regime_VolumeSpike_v1"
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
    
    # 1d HTF data for ATR regime calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d ATR(20) calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar TR
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # first bar TR
    tr3[0] = np.abs(low_1d[0] - close_1d[0])   # first bar TR
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_20_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # 1d ATR(60) for regime comparison (longer term average)
    atr_60_1d = pd.Series(tr).rolling(window=60, min_periods=60).mean().values
    
    # Align ATR values to 4h timeframe
    atr_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_20_1d)
    atr_60_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_60_1d)
    
    # 4h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 60  # Need 60 for ATR(60) + 20 for Donchian + 20 for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(atr_20_1d_aligned[i]) or np.isnan(atr_60_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Regime filter: current ATR > longer term ATR (volatility expansion)
        vol_regime = atr_20_1d_aligned[i] > atr_60_1d_aligned[i]
        
        # Donchian breakout conditions (using prior bar levels to avoid look-ahead)
        breakout_up = curr_close > donchian_high[i-1]  # Break above upper channel
        breakout_down = curr_close < donchian_low[i-1]  # Break below lower channel
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up, volume spike, volatility expansion
            if breakout_up and vol_spike and vol_regime:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down, volume spike, volatility expansion
            elif breakout_down and vol_spike and vol_regime:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Donchian breakdown or volatility contraction
            if curr_close < donchian_low[i] or not vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Donchian breakout or volatility contraction
            if curr_close > donchian_high[i] or not vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-01 18:14
