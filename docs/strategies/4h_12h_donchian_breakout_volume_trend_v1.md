# Strategy: 4h_12h_donchian_breakout_volume_trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.081 | +11.7% | -21.2% | 63 | DISCARD |
| ETHUSDT | 0.194 | +31.6% | -21.9% | 59 | KEEP |
| SOLUSDT | 0.876 | +178.9% | -26.9% | 58 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.106 | +6.7% | -17.9% | 21 | KEEP |
| SOLUSDT | -0.275 | -2.9% | -15.8% | 20 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_donchian_breakout_volume_trend_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop for 12h data (48 bars of 4h)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    # Calculate 12h Donchian channels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian(20) on 12h timeframe
    donch_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: 4h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 1.5 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_up = price_close > donch_high_aligned[i]
        breakout_down = price_close < donch_low_aligned[i]
        
        # Trend filter: price relative to 12h EMA50
        above_trend = price_close > ema50_aligned[i]
        below_trend = price_close < ema50_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Donchian breakout up + above 12h EMA50 + volume confirmation
        if breakout_up and above_trend and vol_confirm:
            enter_long = True
        
        # Short: Donchian breakout down + below 12h EMA50 + volume confirmation
        if breakout_down and below_trend and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Donchian breakout
        exit_long = breakout_down
        exit_short = breakout_up
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 12h Donchian(20) breakouts with trend filter (12h EMA50) and volume confirmation on 4h timeframe.
# Works in both bull and bear markets by only taking breakouts in the direction of the higher timeframe trend.
# Volume confirmation ensures genuine breakouts with participation. Position size 0.25 limits drawdown.
# Target: 20-50 trades per year (80-200 total over 4 years). Uses 12h timeframe for structure, 4h for execution.
```

## Last Updated
2026-04-13 08:56
