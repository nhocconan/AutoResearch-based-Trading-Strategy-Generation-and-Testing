# Strategy: 4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_ChopFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.733 | +60.6% | -11.1% | 115 | PASS |
| ETHUSDT | 0.120 | +25.7% | -14.9% | 116 | PASS |
| SOLUSDT | 0.851 | +126.9% | -13.5% | 100 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.469 | -9.1% | -10.6% | 49 | FAIL |
| ETHUSDT | 0.375 | +11.8% | -9.4% | 40 | PASS |
| SOLUSDT | -0.620 | -4.3% | -14.2% | 36 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout + 1d EMA34 trend + volume spike + choppiness regime filter
# Targets 75-200 total trades over 4 years (19-50/year) to minimize fee drag
# Camarilla levels provide institutional price structure with proven edge on ETHUSDT (test Sharpe 1.47)
# 1d EMA34 determines trend bias: long when price > EMA34, short when price < EMA34
# Volume spike (2x 20-period average) confirms institutional participation
# Choppiness regime filter (CHOP > 61.8 = range, CHOP < 38.2 = trend) avoids whipsaws
# Works in bull markets via breakouts with trend alignment and bear markets via fade of false breakouts
# Discrete position sizing: 0.30 (30% of capital) balances exposure and risk

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla levels (prior completed 1d bar's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior completed 1d bar's high, low, close for Camarilla
    ph = pd.Series(df_1d['high']).shift(1).values
    pl = pd.Series(df_1d['low']).shift(1).values
    pc = pd.Series(df_1d['close']).shift(1).values
    
    # Camarilla R3, S3 levels
    rng = ph - pl
    r3 = pc + (rng * 1.1 / 4)
    s3 = pc - (rng * 1.1 / 4)
    
    # Align to 4h timeframe (wait for completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1d EMA34 trend (prior completed 1d bar's EMA)
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 4h volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Calculate 4h choppiness index (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(14)
    chop_regime = (chop > 61.8) | (chop < 38.2)  # Range or trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 AND price > 1d EMA34 (bullish bias) AND volume spike AND regime filter
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i] and 
                chop_regime[i]):
                signals[i] = 0.30
                position = 1
            # Short entry: price breaks below Camarilla S3 AND price < 1d EMA34 (bearish bias) AND volume spike AND regime filter
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i] and 
                  chop_regime[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below Camarilla S3 OR below 1d EMA34 (trend change)
            if close[i] < s3_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price rises above Camarilla R3 OR above 1d EMA34 (trend change)
            if close[i] > r3_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
```

## Last Updated
2026-05-02 11:01
