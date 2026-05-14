# Strategy: 4h_Donchian20_EMA34_Trend_VolumeSpike_ATRStop_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.336 | +37.6% | -13.6% | 145 | PASS |
| ETHUSDT | 0.855 | +88.0% | -9.7% | 126 | PASS |
| SOLUSDT | 0.518 | +72.4% | -27.9% | 127 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.237 | +3.4% | -7.0% | 54 | FAIL |
| ETHUSDT | 1.296 | +29.5% | -6.7% | 49 | PASS |
| SOLUSDT | 1.446 | +34.0% | -8.4% | 37 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter, volume spike confirmation, and ATR-based stoploss.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA34 trend filter (price above/below EMA).
- Entry: Long when price breaks above Donchian(20) high AND price > 1d EMA34 AND volume > 2.0 * 4h volume MA(20);
         Short when price breaks below Donchian(20) low AND price < 1d EMA34 AND volume > 2.0 * 4h volume MA(20).
- Exit: Opposite Donchian breakout OR ATR(14) trailing stop (long: highest_high - 2.5*ATR; short: lowest_low + 2.5*ATR).
- Signal size: 0.30 discrete to balance profit potential and fee drag.
- Designed to capture strong trends with volatility-adjusted exits and volume confirmation to avoid false breakouts.
- Works in bull markets via upward breakouts, bear markets via downward breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian and volume MA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Donchian channels (20-period) on 4h
    period20_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) on 4h for stoploss
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Handle first value for roll
    tr.iloc[0] = high_4h[0] - low_4h[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) on 4h
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(34) on 1d
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to primary 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, period20_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, period20_low)
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    # Track extreme prices for trailing stop
    highest_high = 0.0
    lowest_low = float('inf')
    
    # Start from index where all indicators are ready (max of 20 for Donchian, 20 for volume, 14 for ATR, 34 for EMA)
    start_idx = max(20, 20, 14, 34)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(atr_aligned[i]) or np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high = 0.0
                lowest_low = float('inf')
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Update trailing stop extremes
        if position == 1:  # long
            if curr_high > highest_high:
                highest_high = curr_high
        elif position == -1:  # short
            if curr_low < lowest_low:
                lowest_low = curr_low
        
        # Calculate stop levels
        long_stop = highest_high - 2.5 * atr_aligned[i] if highest_high > 0 else 0.0
        short_stop = lowest_low + 2.5 * atr_aligned[i] if lowest_low != float('inf') else float('inf')
        
        # Check for stoploss
        if position == 1 and curr_close < long_stop:
            signals[i] = 0.0
            position = 0
            highest_high = 0.0
            lowest_low = float('inf')
            continue
        elif position == -1 and curr_close > short_stop:
            signals[i] = 0.0
            position = 0
            highest_high = 0.0
            lowest_low = float('inf')
            continue
        
        # Breakout conditions
        bullish_breakout = curr_close > donchian_high_aligned[i]
        bearish_breakout = curr_close < donchian_low_aligned[i]
        
        # Trend filter from 1d EMA34
        price_above_ema = curr_close > ema_34_aligned[i]
        price_below_ema = curr_close < ema_34_aligned[i]
        
        # Volume confirmation
        vol_confirm = curr_volume > 2.0 * vol_ma_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: bullish breakout AND price above 1d EMA34
                if bullish_breakout and price_above_ema:
                    signals[i] = 0.30
                    position = 1
                    highest_high = curr_high
                    lowest_low = float('inf')
                # Short: bearish breakout AND price below 1d EMA34
                elif bearish_breakout and price_below_ema:
                    signals[i] = -0.30
                    position = -1
                    highest_high = 0.0
                    lowest_low = curr_low
        elif position == 1:
            # Long position: exit on bearish breakout (already handled stop above)
            signals[i] = 0.30
        elif position == -1:
            # Short position: exit on bullish breakout (already handled stop above)
            signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_EMA34_Trend_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-24 18:50
