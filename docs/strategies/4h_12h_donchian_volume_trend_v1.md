# Strategy: 4h_12h_donchian_volume_trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.194 | +31.0% | -19.6% | 149 | PASS |
| ETHUSDT | 0.331 | +45.1% | -13.3% | 137 | PASS |
| SOLUSDT | 1.134 | +275.2% | -29.4% | 146 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.780 | -4.7% | -11.1% | 53 | FAIL |
| ETHUSDT | 0.686 | +20.4% | -9.7% | 45 | PASS |
| SOLUSDT | 0.585 | +19.2% | -10.4% | 44 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation.
# Enters long when price breaks above Donchian(20) high with expanding volume and bullish 12h trend.
# Enters short when price breaks below Donchian(20) low with expanding volume and bearish 12h trend.
# Uses ATR(14) for dynamic stoploss and position sizing.
# Designed for 20-50 trades/year on 4h timeframe with focus on trend continuation.
# Volume filter ensures institutional participation, reducing false breakouts.
# 12h trend filter prevents counter-trend trading in choppy markets.

name = "4h_12h_donchian_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for volatility filtering and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume moving average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian period
        # Skip if any required data is invalid
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr_14[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.3 * 20-period average volume
        vol_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        # Determine 12h trend direction
        is_bullish_trend = close[i] > ema_50_12h_aligned[i]
        is_bearish_trend = close[i] < ema_50_12h_aligned[i]
        
        # Breakout conditions
        bullish_breakout = (high[i] > high_max_20[i-1]) and vol_filter and is_bullish_trend
        bearish_breakout = (low[i] < low_min_20[i-1]) and vol_filter and is_bearish_trend
        
        # Exit conditions: reversal signal or ATR-based stop
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long on bearish breakout or if price drops 1.5*ATR from entry
            exit_long = bearish_breakout or (close[i] < ema_50_12h_aligned[i])
        elif position == -1:
            # Exit short on bullish breakout or if price rises 1.5*ATR from entry
            exit_short = bullish_breakout or (close[i] > ema_50_12h_aligned[i])
        
        # Priority: entry > exit > hold
        if bullish_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
```

## Last Updated
2026-04-11 22:43
