# Strategy: 4h_12h_donchian_ema_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.176 | +27.1% | -8.8% | 280 | PASS |
| ETHUSDT | 0.255 | +31.1% | -6.2% | 238 | PASS |
| SOLUSDT | 0.057 | +21.3% | -14.6% | 219 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.065 | -7.8% | -10.1% | 103 | FAIL |
| ETHUSDT | 0.712 | +13.6% | -5.7% | 87 | PASS |
| SOLUSDT | 0.210 | +8.2% | -9.4% | 72 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA(50) trend filter and volume confirmation
# - Uses 4h Donchian channels (20-period) for breakout entries
# - Trend filter: 12h EMA(50) to align with intermediate trend and avoid counter-trend trades
# - Volume confirmation: volume > 2.0x 20-period average to ensure breakout strength
# - ATR(14) trailing stop at 2.0x ATR from extreme for risk control
# - Position size: 0.25 (25% of capital) - discrete level to minimize fee churn
# - Target: ~19-50 trades/year (75-200 total over 4 years) per strategy guidelines
# - Daily EMA filter reduces whipsaws during counter-trend bounces
# - Novelty: Applying proven 4h Donchian+volume concept with 12h EMA filter for better trend alignment

name = "4h_12h_donchian_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Pre-compute 12h indicators
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume > 2.0x 20-period average (strict volume confirmation)
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    # 4h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(atr[i]) or
            atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.0x ATR from high
            if low[i] <= highest_since_entry - (2.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.0x ATR from low
            if high[i] >= lowest_since_entry + (2.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and 12h EMA trend filter
            # Long: price breaks above Donchian upper channel AND price > 12h EMA50 AND volume spike
            if (high[i] >= donchian_high[i] and 
                close[i] > ema_50_12h_aligned[i] and
                volume_spike[i]):
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = high[i]  # Initialize for shorts
                signals[i] = 0.25
            # Short: price breaks below Donchian lower channel AND price < 12h EMA50 AND volume spike
            elif (low[i] <= donchian_low[i] and 
                  close[i] < ema_50_12h_aligned[i] and
                  volume_spike[i]):
                position = -1
                highest_since_entry = low[i]  # Initialize for longs
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 22:35
