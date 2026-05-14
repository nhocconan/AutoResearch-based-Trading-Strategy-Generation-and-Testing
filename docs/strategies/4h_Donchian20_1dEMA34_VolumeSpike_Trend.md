# Strategy: 4h_Donchian20_1dEMA34_VolumeSpike_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.289 | +37.8% | -19.3% | 99 | PASS |
| ETHUSDT | 0.190 | +31.2% | -14.8% | 104 | PASS |
| SOLUSDT | 0.758 | +142.3% | -32.3% | 98 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.866 | -5.2% | -11.9% | 44 | FAIL |
| ETHUSDT | 0.142 | +7.5% | -14.7% | 35 | PASS |
| SOLUSDT | 0.304 | +11.4% | -15.2% | 31 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend + volume spike
# Uses 1d EMA34 to establish primary trend (bullish if close > EMA34, bearish if close < EMA34)
# Enters long when price breaks above 4h Donchian upper channel + volume > 2.0 x 20-period EMA + bullish 1d trend
# Enters short when price breaks below 4h Donchian lower channel + volume > 2.0 x 20-period EMA + bearish 1d trend
# Exits on opposite Donchian breakout or when 1d trend reverses
# Volume spike confirms institutional participation, reducing false breakouts
# Designed for 4h timeframe targeting 20-50 trades/year with discrete sizing (0.30)
# Works in bull markets (breakouts with volume in uptrend) and bear markets (breakouts with volume in downtrend)

name = "4h_Donchian20_1dEMA34_VolumeSpike_Trend"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 4h data for Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    upper_channel = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_channel)
    
    # Get 4h data for volume EMA(20) for volume confirmation
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h volume EMA(20) for volume confirmation
    vol_4h = df_4h['volume'].values
    vol_ema_20 = pd.Series(vol_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 2.0 x 20-period EMA
        volume_confirmed = volume[i] > (2.0 * vol_ema_20_aligned[i])
        
        # 1d trend: bullish if close > EMA34, bearish if close < EMA34
        bullish_trend = close[i] > ema_34_1d_aligned[i]
        bearish_trend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume confirmation + bullish 1d trend
            if (close[i] > upper_aligned[i] and volume_confirmed and bullish_trend):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below lower Donchian + volume confirmation + bearish 1d trend
            elif (close[i] < lower_aligned[i] and volume_confirmed and bearish_trend):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price falls below lower Donchian OR 1d trend turns bearish
            if close[i] < lower_aligned[i] or bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price rises above upper Donchian OR 1d trend turns bullish
            if close[i] > upper_aligned[i] or bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
```

## Last Updated
2026-05-04 03:13
