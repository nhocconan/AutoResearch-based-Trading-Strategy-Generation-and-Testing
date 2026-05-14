# Strategy: 4h_Donchian20_1dEMA_VolumeTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.383 | +46.8% | -16.0% | 206 | PASS |
| ETHUSDT | 0.241 | +36.4% | -18.5% | 248 | PASS |
| SOLUSDT | 0.742 | +147.2% | -33.3% | 320 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.298 | +1.2% | -9.6% | 89 | FAIL |
| ETHUSDT | 0.993 | +29.4% | -9.6% | 71 | PASS |
| SOLUSDT | 0.639 | +21.4% | -12.2% | 73 | PASS |

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
    
    # Get 1d data for Donchian channel and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels on 1d
    upper_channel = np.full_like(close_1d, np.nan)
    lower_channel = np.full_like(close_1d, np.nan)
    
    for i in range(19, len(close_1d)):
        upper_channel[i] = np.max(high_1d[i-19:i+1])
        lower_channel[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate 34-period EMA on 1d for trend filter
    if len(close_1d) >= 34:
        ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    else:
        ema_1d = np.full_like(close_1d, np.nan)
    
    # Calculate ATR on 1d for volatility filter
    def calculate_atr(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        
        # True Range
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder smoothing for ATR
        atr = np.full_like(high, np.nan)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Calculate 20-period volume average on 4h
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    
    vol_ma_4h = np.full_like(volume_4h, np.nan)
    vol_period = 20
    
    if len(volume_4h) >= vol_period:
        for i in range(vol_period, len(volume_4h)):
            vol_ma_4h[i] = np.mean(volume_4h[i-vol_period:i])
    
    # Align all data to 4h timeframe (primary)
    upper_channel_4h = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_4h = align_htf_to_ltf(prices, df_1d, lower_channel)
    ema_1d_4h = align_htf_to_ltf(prices, df_1d, ema_1d)
    atr_1d_4h = align_htf_to_ltf(prices, df_1d, atr_1d)
    vol_ma_4h_4h = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(19, 34, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel_4h[i]) or np.isnan(lower_channel_4h[i]) or 
            np.isnan(ema_1d_4h[i]) or np.isnan(atr_1d_4h[i]) or 
            np.isnan(vol_ma_4h_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average (4h)
        vol_confirm = volume[i] > 1.5 * vol_ma_4h_4h[i]
        
        # Trend filter: price above/below EMA
        uptrend = close[i] > ema_1d_4h[i]
        downtrend = close[i] < ema_1d_4h[i]
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_1d_4h[i] > 0.01 * close[i]  # ATR > 1% of price
        
        if position == 0:
            # Long: price breaks above upper Donchian channel with uptrend and volume
            if close[i] > upper_channel_4h[i] and uptrend and vol_confirm and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian channel with downtrend and volume
            elif close[i] < lower_channel_4h[i] and downtrend and vol_confirm and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below lower Donchian channel OR trend reverses
            if close[i] < lower_channel_4h[i] or not uptrend:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above upper Donchian channel OR trend reverses
            if close[i] > upper_channel_4h[i] or not downtrend:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA_VolumeTrend"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 18:22
