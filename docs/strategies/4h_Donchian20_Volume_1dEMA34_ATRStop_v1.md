# Strategy: 4h_Donchian20_Volume_1dEMA34_ATRStop_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.193 | +29.5% | -11.3% | 183 | PASS |
| ETHUSDT | 0.074 | +22.7% | -12.4% | 178 | PASS |
| SOLUSDT | 0.459 | +65.6% | -26.4% | 189 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.763 | -9.8% | -10.6% | 71 | FAIL |
| ETHUSDT | 0.580 | +14.6% | -8.7% | 60 | PASS |
| SOLUSDT | 0.441 | +13.1% | -10.6% | 59 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d EMA34 trend filter + ATR-based stoploss
# Long when price breaks above 20-bar Donchian high with volume > 1.5x 24-bar average and close > 1d EMA34 (uptrend)
# Short when price breaks below 20-bar Donchian low with volume > 1.5x 24-bar average and close < 1d EMA34 (downtrend)
# Exit via ATR trailing stop: long exit when price < highest_high_since_entry - 2.0 * ATR, short exit when price > lowest_low_since_entry + 2.0 * ATR
# Donchian channels provide clear structure, volume confirms conviction, 1d EMA34 filters for higher-timeframe trend, ATR stop manages risk.
# Target: 75-200 total trades over 4 years = 19-50/year. Uses discrete sizing (0.30) to balance return and fee drag.

name = "4h_Donchian20_Volume_1dEMA34_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (1.5x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(20, 34, 24, 14) + 1  # Donchian(20) + EMA34(1d) + volume MA(24) + ATR(14) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high with volume spike and close > 1d EMA34 (uptrend)
            if (close[i] > donchian_high[i] and 
                volume_spike[i] and close[i] > ema_34_aligned[i]):
                signals[i] = 0.30
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
            # Short entry: price breaks below Donchian low with volume spike and close < 1d EMA34 (downtrend)
            elif (close[i] < donchian_low[i] and 
                  volume_spike[i] and close[i] < ema_34_aligned[i]):
                signals[i] = -0.30
                position = -1
                entry_bar = i
                lowest_since_entry = low[i]
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # ATR trailing stop: exit when price < highest_high_since_entry - 2.0 * ATR
            if close[i] < highest_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # ATR trailing stop: exit when price > lowest_low_since_entry + 2.0 * ATR
            if close[i] > lowest_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
```

## Last Updated
2026-05-03 01:50
