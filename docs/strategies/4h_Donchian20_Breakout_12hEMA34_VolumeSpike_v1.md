# Strategy: 4h_Donchian20_Breakout_12hEMA34_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.078 | +23.5% | -12.9% | 140 | PASS |
| ETHUSDT | 0.504 | +51.5% | -10.3% | 132 | PASS |
| SOLUSDT | 0.750 | +106.9% | -21.9% | 119 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.815 | -2.2% | -8.5% | 58 | FAIL |
| ETHUSDT | 0.889 | +20.6% | -7.4% | 46 | PASS |
| SOLUSDT | 0.693 | +17.1% | -8.1% | 38 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 12h EMA34 Trend + Volume Spike + ATR Stop
Hypothesis: Donchian channel breakouts capture strong momentum. 12h EMA34 filters for trend alignment (works in bull/bear via trend filter). Volume spike confirms breakout strength. ATR-based trailing stop manages risk. Discrete sizing (0.25) limits fee drag. Target: 20-50 trades/year.
"""

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
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate ATR for stop loss (using 20 periods)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for ATR (20) and EMA34 (34)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_34_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_aligned[i]
        atr_value = atr[i]
        
        # Donchian channels (20-period) - using lookback excluding current bar
        if i >= 20:
            donchian_high = np.max(high[i-20:i])  # highest high of prior 20 bars
            donchian_low = np.min(low[i-20:i])   # lowest low of prior 20 bars
        else:
            donchian_high = np.max(high[:i]) if i > 0 else high[i]
            donchian_low = np.min(low[:i]) if i > 0 else low[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout conditions: price breaks above/below Donchian channels
        bullish_breakout = curr_close > donchian_high
        bearish_breakout = curr_close < donchian_low
        
        # Update tracking variables for trailing stop logic
        if position == 1:
            highest_since_entry = max(highest_since_entry, curr_high)
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, curr_low)
        
        # Exit conditions: trailing stop or reverse breakout
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Trailing stop: exit if price drops 3.0*ATR from highest since entry
                if curr_close < highest_since_entry - 3.0 * atr_value:
                    exit_signal = True
                # Reverse breakout or trend rejection
                elif curr_close < donchian_low or curr_close < ema_trend:
                    exit_signal = True
                    
            elif position == -1:
                # Trailing stop: exit if price rises 3.0*ATR from lowest since entry
                if curr_close > lowest_since_entry + 3.0 * atr_value:
                    exit_signal = True
                # Reverse breakout or trend rejection
                elif curr_close > donchian_high or curr_close > ema_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
        
        # Entry conditions: Donchian breakout + trend alignment + volume spike
        if position == 0:
            # Long: break above Donchian high AND price above 12h EMA34
            long_condition = bullish_breakout and (curr_close > ema_trend) and volume_spike
            # Short: break below Donchian low AND price below 12h EMA34
            short_condition = bearish_breakout and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            elif short_condition:
                signals[i] = -0.25
                position = -1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 00:54
