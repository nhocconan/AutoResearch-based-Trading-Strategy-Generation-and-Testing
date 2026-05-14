# Strategy: 4h_Camarilla_R1_S1_Breakout_1dATR_Trend_VolumeSp_V3

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.106 | +24.8% | -14.4% | 113 | PASS |
| ETHUSDT | 0.055 | +22.1% | -10.9% | 97 | PASS |
| SOLUSDT | 1.145 | +178.4% | -18.9% | 80 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.262 | -5.2% | -9.3% | 46 | FAIL |
| ETHUSDT | 0.343 | +10.7% | -9.4% | 37 | PASS |
| SOLUSDT | 0.235 | +9.1% | -9.5% | 25 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dATR_Trend_VolumeSp_V3
Hypothesis: Further tighten volume confirmation to 2.5x average and add ATR-based volatility filter (ATR > 0.5 * ATR_ma) to ensure breakouts occur only during sufficient volatility. This should reduce false breakouts in low-volatility periods while maintaining edge in both bull and bear markets. Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots, EMA34, ATR regime filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Camarilla pivot levels (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    R1 = prev_close + 0.5 * prev_range
    S1 = prev_close - 0.5 * prev_range
    
    # Align 1d pivot levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume spike: current volume > 2.5 * 20-period average (tighter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.5)
    
    # 1d ATR for volatility regime filter (loaded ONCE)
    tr1 = df_1d['high'].values - df_1d['low'].values
    tr2 = np.abs(df_1d['high'].values - df_1d['close'].shift(1).values)
    tr3 = np.abs(df_1d['low'].values - df_1d['close'].shift(1).values)
    tr_daily = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_daily).rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    volatility_filter = atr_1d > (atr_ma_1d * 0.5)  # Only trade when volatility is above 50% of MA
    
    # Align volatility filter to 4h timeframe
    volatility_filter_aligned = align_htf_to_ltf(prices, df_1d, volatility_filter)
    
    # 1d EMA34 for trend filter (loaded ONCE)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # ATR for stoploss (using 4h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d indicators (34 for EMA, 20 for ATR MA, 14 for ATR)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(volatility_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals with volume spike, volatility filter, and trend alignment
            # Long breakout: price breaks above R1 with uptrend, volume spike, and sufficient volatility
            long_breakout = (curr_close > R1_aligned[i]) and uptrend and volume_spike[i] and volatility_filter_aligned[i]
            # Short breakout: price breaks below S1 with downtrend, volume spike, and sufficient volatility
            short_breakout = (curr_close < S1_aligned[i]) and downtrend and volume_spike[i] and volatility_filter_aligned[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Stoploss: 2.0 * ATR below entry
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks below S1 (mean reversion) or trend changes
            elif curr_close < S1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Stoploss: 2.0 * ATR above entry
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks above R1 (mean reversion) or trend changes
            elif curr_close > R1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dATR_Trend_VolumeSp_V3"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 09:37
