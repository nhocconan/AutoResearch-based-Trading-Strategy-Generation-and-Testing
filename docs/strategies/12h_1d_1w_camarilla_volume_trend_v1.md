# Strategy: 12h_1d_1w_camarilla_volume_trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.369 | +42.4% | -10.2% | 12 | PASS |
| ETHUSDT | 0.182 | +30.4% | -21.5% | 16 | PASS |
| SOLUSDT | 0.854 | +146.6% | -31.9% | 16 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.108 | +7.0% | -7.9% | 12 | PASS |
| ETHUSDT | -0.193 | +4.0% | -6.9% | 5 | FAIL |
| SOLUSDT | 0.350 | +10.8% | -7.6% | 5 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and 1w trend filter
    # Long: price breaks above H3 level + volume > 1.3x 20-period average + 1w close > 1w EMA20
    # Short: price breaks below L3 level + volume > 1.3x 20-period average + 1w close < 1w EMA20
    # Uses discrete sizing (0.25) to minimize fee drag and ATR-based stoploss
    # Target: 12-37 trades/year to stay within 12h optimal range
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Get 1d data for Camarilla pivots and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # Pivot = (H+L+C)/3
    # H3 = Pivot + 1.1*(H-L)
    # L3 = Pivot - 1.1*(H-L)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    hl_range = high_1d - low_1d
    h3 = pivot + 1.1 * hl_range
    l3 = pivot - 1.1 * hl_range
    
    # Calculate 1d volume average (20-period) for confirmation
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA20 for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    atr_1d = np.zeros(n)  # Simplified ATR using daily range
    
    # Calculate simplified ATR (daily range) for stoploss
    for i in range(n):
        idx_1d = i // 2  # 12h bars in 1d timeframe (2 bars per day)
        if idx_1d < len(high_1d) and idx_1d < len(low_1d):
            daily_range = high_1d[idx_1d] - low_1d[idx_1d]
            atr_1d[i] = daily_range * 0.5  # Approximate ATR as 50% of daily range
        else:
            atr_1d[i] = 0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        idx_1d = i // 2  # 12h bars in 1d timeframe (2 bars per day)
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_1d[idx_1d] > 1.3 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: 1w close above/below EMA20
        uptrend = close_1d[idx_1d] > ema_20_1w_aligned[i] if idx_1d < len(close_1d) else False
        downtrend = close_1d[idx_1d] < ema_20_1w_aligned[i] if idx_1d < len(close_1d) else False
        
        # Breakout conditions: price breaks Camarilla levels with volume and trend
        breakout_long = (close[i] > h3_aligned[i]) and volume_confirmed and uptrend
        breakout_short = (close[i] < l3_aligned[i]) and volume_confirmed and downtrend
        
        # Stoploss: 1.5x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 1.5 * atr_1d[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 1.5 * atr_1d[i]
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "12h_1d_1w_camarilla_volume_trend_v1"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-13 09:48
