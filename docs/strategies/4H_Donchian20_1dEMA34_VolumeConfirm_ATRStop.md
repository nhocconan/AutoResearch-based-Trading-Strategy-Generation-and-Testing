# Strategy: 4H_Donchian20_1dEMA34_VolumeConfirm_ATRStop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.332 | +37.2% | -11.4% | 150 | KEEP |
| ETHUSDT | 0.483 | +52.7% | -12.6% | 143 | KEEP |
| SOLUSDT | 0.788 | +118.2% | -22.2% | 139 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.120 | -2.1% | -5.9% | 72 | DISCARD |
| ETHUSDT | 1.058 | +23.0% | -6.4% | 57 | KEEP |
| SOLUSDT | 1.067 | +21.8% | -5.5% | 51 | KEEP |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter, volume confirmation, and ATR-based stoploss.
Long when price breaks above Donchian upper band AND close > 1d EMA34 AND volume > 1.8x 20-period average.
Short when price breaks below Donchian lower band AND close < 1d EMA34 AND volume > 1.8x 20-period average.
Exit when price crosses the Donchian middle band (20-period SMA of (high+low)/2).
Uses discrete position sizing (0.28) to balance profit potential and risk. Targets 20-50 trades/year per symbol.
The 1d EMA34 provides a robust trend filter that adapts to both bull and bear markets, avoiding counter-trend entries.
Volume confirmation at 1.8x ensures only high-momentum breakouts are taken, reducing false signals and trade frequency.
ATR stoploss is implemented via signal=0 when adverse price movement exceeds 2.5x ATR(20) from entry.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for price action - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d data
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate ATR(20) for stoploss on 4h data
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period: no previous close
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels on 4h data
    # Upper band: 20-period high
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    # Middle band: 20-period SMA of (high+low)/2
    hl_avg = (high_4h + low_4h) / 2
    donchian_middle = pd.Series(hl_avg).rolling(window=20, min_periods=20).mean().values
    
    # Align Donchian levels to 4h timeframe (already on 4h, but align for safety)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    max_favorable_price = 0.0  # For ATR trailing stop logic
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                max_favorable_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND close > 1d EMA34 AND volume spike
            if (price > donchian_upper_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.28
                position = 1
                entry_price = price
                max_favorable_price = price
            # Short: price breaks below Donchian lower band AND close < 1d EMA34 AND volume spike
            elif (price < donchian_lower_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.28
                position = -1
                entry_price = price
                max_favorable_price = price
        else:
            # Update max favorable price for trailing stop
            if position == 1:
                if price > max_favorable_price:
                    max_favorable_price = price
            else:  # position == -1
                if price < max_favorable_price:
                    max_favorable_price = price
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses Donchian middle band
            if position == 1 and price < donchian_middle_aligned[i]:
                exit_signal = True
            elif position == -1 and price > donchian_middle_aligned[i]:
                exit_signal = True
            
            # ATR-based stoploss: exit if adverse move > 2.5 * ATR from max favorable price
            if not exit_signal:
                if position == 1:
                    adverse_move = max_favorable_price - price
                    if adverse_move > 2.5 * atr_val:
                        exit_signal = True
                else:  # position == -1
                    adverse_move = price - max_favorable_price
                    if adverse_move > 2.5 * atr_val:
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                max_favorable_price = 0.0
            else:
                signals[i] = 0.28 if position == 1 else -0.28
    
    return signals

name = "4H_Donchian20_1dEMA34_VolumeConfirm_ATRStop"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 04:26
