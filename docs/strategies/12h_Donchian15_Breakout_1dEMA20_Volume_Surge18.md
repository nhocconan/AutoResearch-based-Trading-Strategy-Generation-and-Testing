# Strategy: 12h_Donchian15_Breakout_1dEMA20_Volume_Surge18

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.447 | -10.7% | -24.4% | 214 | FAIL |
| ETHUSDT | 0.312 | +43.9% | -17.1% | 213 | PASS |
| SOLUSDT | 0.877 | +187.7% | -25.8% | 226 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.351 | +12.7% | -10.7% | 76 | PASS |
| SOLUSDT | 0.289 | +11.4% | -12.1% | 70 | PASS |

## Code
```python
#!/usr/bin/env python3
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
    
    # Get 1d data once for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Donchian channel (15) - tighter for fewer signals
    donchian_high = pd.Series(high_1d).rolling(window=15, min_periods=15).max().values
    donchian_low = pd.Series(low_1d).rolling(window=15, min_periods=15).min().values
    
    # 1d EMA20 - trend confirmation
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1d ATR10 - volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Align HTF indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    atr_10_aligned = align_htf_to_ltf(prices, df_1d, atr_10)
    
    # Volume surge: current volume > 1.8x 15-period average (12h)
    vol_ma_15 = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
    volume_surge = volume > (vol_ma_15 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(atr_10_aligned[i]) or 
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high_aligned[i]
        breakout_down = close[i] < donchian_low_aligned[i]
        
        # Trend filter: price above/below 1d EMA20
        trend_up = close[i] > ema_20_1d_aligned[i]
        trend_down = close[i] < ema_20_1d_aligned[i]
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr_10_aligned[i] > 0.006 * close[i]  # ATR > 0.6% of price
        
        # Entry conditions - stricter criteria to reduce trade frequency
        # Long: upward breakout + uptrend + volume surge + vol filter
        long_entry = breakout_up and trend_up and volume_surge[i] and vol_filter
        # Short: downward breakout + downtrend + volume surge + vol filter
        short_entry = breakout_down and trend_down and volume_surge[i] and vol_filter
        
        # Exit conditions: opposite breakout or trend reversal
        long_exit = breakout_down or not trend_up
        short_exit = breakout_up or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian15_Breakout_1dEMA20_Volume_Surge18"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-28 06:02
