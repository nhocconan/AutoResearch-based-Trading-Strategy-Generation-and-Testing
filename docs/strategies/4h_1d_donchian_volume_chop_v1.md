# Strategy: 4h_1d_donchian_volume_chop_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.493 | +8.6% | -8.7% | 72 | FAIL |
| ETHUSDT | 0.460 | +38.9% | -6.6% | 72 | PASS |
| SOLUSDT | 0.082 | +23.6% | -10.0% | 64 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.083 | +6.7% | -4.8% | 32 | PASS |
| SOLUSDT | -0.415 | +2.4% | -6.7% | 22 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d chop regime filter
# - Long when price breaks above Donchian(20) high AND volume > 1.5x 20-period average AND 1d chop > 61.8 (range)
# - Short when price breaks below Donchian(20) low AND volume > 1.5x 20-period average AND 1d chop > 61.8 (range)
# - Exit when price returns to Donchian(20) middle or opposite signal
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Works in range markets via mean reversion at Donchian extremes with volume confirmation
# - Chop filter ensures we only trade in ranging conditions where breakouts are more likely to fail

name = "4h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h Donchian channels (20)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_high + donchian_low) / 2
    
    # Pre-compute 4h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d chop regime (choppiness index)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - np.roll(close_1d, 1)[1:])
    tr3 = np.abs(low_1d[1:] - np.roll(close_1d, 1)[1:])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # first element is NaN
    
    # ATR(14)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # Max(high) - Min(low) over 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_max_min = max_high - min_low
    
    # Chop = 100 * log10(tr_sum / range_max_min) / log10(14)
    chop = 100 * np.log10(tr_sum / range_max_min) / np.log10(14)
    chop = np.concatenate([np.full(13, np.nan), chop[13:]])  # align indices
    
    # Chop regime: > 61.8 = ranging (good for mean reversion at extremes)
    chop_range = chop > 61.8
    
    # Align HTF indicators to 4h timeframe
    chop_range_aligned = align_htf_to_ltf(prices, df_1d, chop_range)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop_range_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Donchian high AND volume spike AND chop range
            if (close[i] > donchian_high[i-1] and  # breakout above previous period's high
                volume_spike[i] and 
                chop_range_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Donchian low AND volume spike AND chop range
            elif (close[i] < donchian_low[i-1] and  # breakout below previous period's low
                  volume_spike[i] and 
                  chop_range_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price returns to Donchian middle or opposite signal with volume
            exit_long = (position == 1 and 
                        (close[i] <= donchian_middle[i] or
                         (close[i] < donchian_low[i] and volume_spike[i])))
            exit_short = (position == -1 and 
                         (close[i] >= donchian_middle[i] or
                          (close[i] > donchian_high[i] and volume_spike[i])))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-10 07:20
