# Strategy: 6h_Donchian20_1dEMA50_VolumeSpike_ATRStop_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.078 | +18.1% | -8.5% | 62 | FAIL |
| ETHUSDT | 0.074 | +23.4% | -8.7% | 50 | PASS |
| SOLUSDT | 0.493 | +62.2% | -23.6% | 60 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.502 | +12.5% | -6.0% | 23 | PASS |
| SOLUSDT | 0.072 | +6.5% | -7.6% | 22 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation.
# Uses ATR-based trailing stop for risk management. Discrete sizing 0.25.
# Target: 50-150 total trades over 4 years (12-37/year).
# Donchian channels provide clear breakout levels that work in both bull and bear markets.
# 1d EMA50 filter ensures we only trade in the direction of the higher timeframe trend.
# Volume confirmation reduces false breakouts.

name = "6h_Donchian20_1dEMA50_VolumeSpike_ATRStop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channels (20-period) from prior completed 1d bar
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:  # Need at least 20 periods for Donchian + 1 for prior
        return np.zeros(n)
    
    # Calculate 20-period Donchian high/low on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Rolling max/min for 20-period Donchian
    donchian_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only prior completed 1d bar (avoid look-ahead)
    prior_donchian_high = np.roll(donchian_high_1d, 1)
    prior_donchian_high[0] = np.nan
    prior_donchian_low = np.roll(donchian_low_1d, 1)
    prior_donchian_low[0] = np.nan
    
    # Align Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, prior_donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, prior_donchian_low)
    
    # Calculate 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(24) for stoploss (using 6h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=24, min_periods=24, adjust=False).mean().values
    
    # Volume confirmation: volume > 2.0x 24-bar average (on 6h data)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0
    lowest_low_since_entry = 0
    
    for i in range(100, n):  # Start after sufficient warmup
        # Get current values
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if np.isnan(donchian_high_val) or np.isnan(donchian_low_val) or np.isnan(ema_trend) or np.isnan(atr_val):
            continue
            
        # Entry conditions
        # Long: break above prior 1d Donchian high with volume spike and above 1d EMA50
        long_entry = (close[i] > donchian_high_val) and (close[i] > ema_trend) and vol_spike
        # Short: break below prior 1d Donchian low with volume spike and below 1d EMA50
        short_entry = (close[i] < donchian_low_val) and (close[i] < ema_trend) and vol_spike
        
        # Exit conditions (trailing stop)
        long_exit = False
        short_exit = False
        
        if position == 1:  # Long position
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            long_exit = close[i] < (highest_high_since_entry - 2.5 * atr_val)
        elif position == -1:  # Short position
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            short_exit = close[i] > (lowest_low_since_entry + 2.5 * atr_val)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-03 03:35
