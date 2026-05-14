# Strategy: 4h_Donchian20_1dEMA50_VolumeSpike_ATRStop_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.286 | +32.0% | -8.2% | 119 | KEEP |
| ETHUSDT | 0.153 | +27.3% | -14.1% | 112 | KEEP |
| SOLUSDT | 0.579 | +71.1% | -19.4% | 110 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.462 | +2.5% | -4.2% | 42 | DISCARD |
| ETHUSDT | 1.345 | +25.6% | -6.2% | 42 | KEEP |
| SOLUSDT | 0.716 | +16.3% | -5.9% | 36 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Uses ATR(24) trailing stop for risk management. Discrete sizing 0.25 to balance return and fee drag.
# Target: 75-200 total trades over 4 years (19-50/year). Works in bull via breakouts, in bear via short signals.
# Proven pattern from top performers: price channel + HTF trend + volume confirmation + ATR stop.
# Focus on 4h timeframe to reduce trade frequency vs lower timeframes while maintaining sufficient sample size.

name = "4h_Donchian20_1dEMA50_VolumeSpike_ATRStop_v1"
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
    
    # Calculate 4h Donchian channels (20-period) from prior completed 4h bar
    # We need to use the prior completed bar's high/low to avoid look-ahead
    prior_high = np.roll(high, 1)
    prior_high[0] = np.nan
    prior_low = np.roll(low, 1)
    prior_low[0] = np.nan
    
    # Calculate rolling max/min on prior bars to get Donchian levels
    # Using pandas rolling on the prior series to avoid look-ahead
    prior_high_series = pd.Series(prior_high)
    prior_low_series = pd.Series(prior_low)
    donchian_high = prior_high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = prior_low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(24) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=24, min_periods=24, adjust=False).mean().values
    
    # Volume confirmation: volume > 2.0x 24-bar average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0
    lowest_low_since_entry = 0
    
    for i in range(100, n):  # Start after sufficient warmup
        # Get current values
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if np.isnan(donchian_high_val) or np.isnan(donchian_low_val) or np.isnan(ema_trend) or np.isnan(atr_val):
            continue
            
        # Entry conditions
        # Long: break above Donchian high with volume spike and above 1d EMA50
        long_entry = (close[i] > donchian_high_val) and (close[i] > ema_trend) and vol_spike
        # Short: break below Donchian low with volume spike and below 1d EMA50
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
2026-05-03 03:29
