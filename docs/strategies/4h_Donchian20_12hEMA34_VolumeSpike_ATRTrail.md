# Strategy: 4h_Donchian20_12hEMA34_VolumeSpike_ATRTrail

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.407 | +3.3% | -17.4% | 170 | FAIL |
| ETHUSDT | 0.530 | +53.1% | -13.7% | 152 | PASS |
| SOLUSDT | 0.564 | +73.6% | -26.3% | 142 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.485 | +13.1% | -6.8% | 56 | PASS |
| SOLUSDT | 0.415 | +12.2% | -13.6% | 47 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + volume spike + 12h EMA34 trend filter + ATR trailing stop
- Donchian breakout on 4h captures structural momentum moves with proven frequency
- Volume spike (>2.0x 20-period average) confirms breakout strength and reduces false signals
- 12h EMA34 filter ensures alignment with higher timeframe trend to avoid counter-trend trades
- ATR-based trailing stop (3.0 * ATR from extreme) manages risk while allowing trends to run
- Position sizing: 0.25 discrete to minimize fee churn
- Target: 25-50 trades/year per symbol (~100-200 total over 4 years)
- Works in bull markets (captures breakouts) and bear markets (short breakdowns with trend filter)
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
    
    # Get 4h data for primary timeframe calculations
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels (20-period) on 4h
    def calculate_donchian(high_arr, low_arr, window):
        upper = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high_4h, low_4h, 20)
    
    # Calculate EMA34 on 12h for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume average (20-period) on 4h
    volume_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss on 4h
    def calculate_atr(high_arr, low_arr, close_arr, window):
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        return atr
    
    atr_4h = calculate_atr(high_4h, low_4h, close_4h, 14)
    
    # Align all indicators to 4h timeframe
    donch_upper_aligned = align_htf_to_ltf(prices, df_4h, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_4h, donch_lower)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    long_high = 0.0   # track highest close since entry for trailing stop
    low_low = 0.0     # track lowest close since entry for trailing stop
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(donch_upper_aligned[i]) or np.isnan(donch_lower_aligned[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = donch_upper_aligned[i]
        lower = donch_lower_aligned[i]
        ema_trend = ema34_12h_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend alignment
            # Long: price breaks above upper Donchian + volume spike + price > 12h EMA34 (uptrend)
            if price > upper and vol > 2.0 * vol_ma and price > ema_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
                long_high = price
            # Short: price breaks below lower Donchian + volume spike + price < 12h EMA34 (downtrend)
            elif price < lower and vol > 2.0 * vol_ma and price < ema_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
                low_low = price
        
        elif position == 1:
            # Update highest close since entry
            if price > long_high:
                long_high = price
            
            # Exit conditions for long
            exit_signal = False
            
            # Exit 1: ATR trailing stop (3.0 * ATR from highest close)
            if price < long_high - 3.0 * atr_val:
                exit_signal = True
            
            # Exit 2: Price retrace to middle of Donchian channel
            elif price < (upper + lower) / 2:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest close since entry
            if price < low_low:
                low_low = price
            
            # Exit conditions for short
            exit_signal = False
            
            # Exit 1: ATR trailing stop (3.0 * ATR from lowest close)
            if price > low_low + 3.0 * atr_val:
                exit_signal = True
            
            # Exit 2: Price retrace to middle of Donchian channel
            elif price > (upper + lower) / 2:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA34_VolumeSpike_ATRTrail"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-17 21:08
