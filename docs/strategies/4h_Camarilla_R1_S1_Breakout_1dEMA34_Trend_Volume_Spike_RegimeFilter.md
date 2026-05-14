# Strategy: 4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume_Spike_RegimeFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.324 | +35.9% | -12.4% | 343 | PASS |
| ETHUSDT | 0.321 | +38.0% | -10.6% | 311 | PASS |
| SOLUSDT | 0.341 | +46.8% | -16.8% | 264 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.125 | +4.6% | -7.6% | 128 | FAIL |
| ETHUSDT | 1.619 | +33.7% | -10.1% | 113 | PASS |
| SOLUSDT | 0.685 | +17.2% | -7.6% | 93 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume_Spike_RegimeFilter
Hypothesis: Camarilla R1/S1 breakouts with 1d EMA34 trend filter, volume spike confirmation, and choppiness regime filter on 4h timeframe.
Only takes trades when market is not too choppy (CHOP < 61.8) to avoid whipsaws in ranging markets.
Long when price breaks above R1 with uptrend (close > 1d EMA34) and volume > 2.0x average and CHOP < 61.8.
Short when price breaks below S1 with downtrend (close < 1d EMA34) and volume > 2.0x average and CHOP < 61.8.
Uses ATR-based trailing stop (2.0x ATR from extreme) to manage risk.
Designed for moderate trade frequency (20-50/year) to balance opportunity and fee drag.
Uses discrete position sizing (0.30) to minimize fee churn while capturing meaningful moves.
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
    
    # Get 1d data for EMA34 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous 1d bar's high, low, close for Camarilla levels (using 1d data)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels: R1, S1
    camarilla_range = prev_high - prev_low
    R1 = prev_close + camarilla_range * 1.0/12
    S1 = prev_close - camarilla_range * 1.0/12
    
    # Align Camarilla levels to 4h timeframe (1d -> 4h)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: 2.0x average volume (balanced to reduce trades but keep opportunities)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for trailing stop (using 14-period ATR)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index regime filter (14-period)
    # CHOP = 100 * log10(sum(atr14) / (max(high14) - min(low14))) / log10(14)
    # We calculate it on 4h data directly
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values  # same as above atr
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_14 * 14 / (max_high_14 - min_low_14 + 1e-10)) / np.log10(14)
    # Handle division by zero or invalid values
    chop_raw = np.where((max_high_14 - min_low_14) <= 0, 100, chop_raw)
    chop_raw = np.where(np.isnan(chop_raw) | np.isinf(chop_raw), 50, chop_raw)  # neutral if invalid
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Warmup: max of 1d EMA (34), volume MA (20), ATR (14), CHOP (14)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(chop_raw[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        chop_val = chop_raw[i]
        
        # Regime filter: only trade when not too choppy (CHOP < 61.8 = trending market)
        regime_filter = chop_val < 61.8
        
        if position == 0:
            # Long: break above R1, uptrend, volume spike, good regime
            long_signal = (high_val > R1_val) and (close_val > ema_34_1d_val) and (volume_val > 2.0 * vol_ma_val) and regime_filter
            # Short: break below S1, downtrend, volume spike, good regime
            short_signal = (low_val < S1_val) and (close_val < ema_34_1d_val) and (volume_val > 2.0 * vol_ma_val) and regime_filter
            
            if long_signal:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.0 * atr_val
            elif short_signal:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.0 * atr_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.0 * atr_val)
            # Exit: trailing stop hit or trend reversal (price < EMA34) or regime becomes too choppy
            if (low_val < long_stop) or (close_val < ema_34_1d_val) or (chop_val >= 61.8):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.0 * atr_val)
            # Exit: trailing stop hit or trend reversal (price > EMA34) or regime becomes too choppy
            if (high_val > short_stop) or (close_val > ema_34_1d_val) or (chop_val >= 61.8):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume_Spike_RegimeFilter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-26 01:24
