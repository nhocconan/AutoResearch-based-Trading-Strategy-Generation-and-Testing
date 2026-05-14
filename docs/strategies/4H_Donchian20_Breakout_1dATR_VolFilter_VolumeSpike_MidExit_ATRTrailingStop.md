# Strategy: 4H_Donchian20_Breakout_1dATR_VolFilter_VolumeSpike_MidExit_ATRTrailingStop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.936 | +6.2% | -4.7% | 21 | FAIL |
| ETHUSDT | 0.314 | +31.1% | -5.0% | 23 | PASS |
| SOLUSDT | -0.439 | +4.3% | -13.8% | 27 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.095 | +16.5% | -3.8% | 10 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 1d ATR volatility filter and volume confirmation.
Long when price breaks above Donchian(20) high AND ATR(14) < 0.8 * ATR(50) (low volatility regime) AND volume > 1.5x 20-period average.
Short when price breaks below Donchian(20) low AND ATR(14) < 0.8 * ATR(50) AND volume > 1.5x 20-period average.
Exit when price retraces to Donchian midpoint or ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.25) to minimize fee drag. Targets 15-30 trades/year per symbol.
Donchian provides objective breakout levels; low volatility filter avoids choppy markets; volume confirms breakout strength.
Works in bull (breakouts with volume) and bear (breakdowns with volume) markets by capturing expansion phases after low volatility.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR(14) and ATR(50) for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50_1d = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align ATR arrays to 4h timeframe
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    atr50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr50_1d)
    
    # Calculate Donchian(20) channels from 1d OHLC
    # Upper = 20-period high, Lower = 20-period low
    donch_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_mid_1d = (donch_high_1d + donch_low_1d) / 2.0
    
    # Align Donchian levels to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid_1d)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation (using 4h data)
    tr1_4h = np.abs(high - low)
    tr2_4h = np.abs(high - np.roll(close, 1))
    tr3_4h = np.abs(low - np.roll(close, 1))
    tr1_4h[0] = 0
    tr2_4h[0] = 0
    tr3_4h[0] = 0
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 20)  # ATR50 needs 50, Donchian needs 20, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr14_1d_aligned[i]) or np.isnan(atr50_1d_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_4h_val = atr_4h[i]
        atr14_val = atr14_1d_aligned[i]
        atr50_val = atr50_1d_aligned[i]
        donch_high = donch_high_aligned[i]
        donch_low = donch_low_aligned[i]
        donch_mid = donch_mid_aligned[i]
        
        # Volatility filter: low volatility regime (ATR14 < 0.8 * ATR50)
        vol_filter = atr14_val < 0.8 * atr50_val
        
        if position == 0:
            # Long: Break above Donchian high AND low volatility AND volume spike
            if close[i] > donch_high and vol_filter and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below Donchian low AND low volatility AND volume spike
            elif close[i] < donch_low and vol_filter and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to Donchian midpoint
            if position == 1 and close[i] <= donch_mid:
                exit_signal = True
            elif position == -1 and close[i] >= donch_mid:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_4h_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_4h_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dATR_VolFilter_VolumeSpike_MidExit_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 06:52
