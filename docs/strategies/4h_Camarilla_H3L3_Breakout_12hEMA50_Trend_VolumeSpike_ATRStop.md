# Strategy: 4h_Camarilla_H3L3_Breakout_12hEMA50_Trend_VolumeSpike_ATRStop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.312 | +38.6% | -12.4% | 107 | PASS |
| ETHUSDT | 0.109 | +24.4% | -17.0% | 98 | PASS |
| SOLUSDT | 1.105 | +218.3% | -21.2% | 86 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.704 | -3.0% | -8.2% | 42 | FAIL |
| ETHUSDT | 0.561 | +16.9% | -7.6% | 30 | PASS |
| SOLUSDT | 0.070 | +5.9% | -13.2% | 30 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h Camarilla H3L3 Breakout + 12h EMA50 Trend + Volume Spike + ATR Stop
Hypothesis: Camarilla H3/L3 levels act as strong intraday support/resistance. Breakout above H3 or below L3 with volume confirmation and 12h EMA50 trend filter captures momentum moves. ATR-based stoploss manages risk. Works in bull (long on H3 break) and bear (short on L3 break). Volume spike ensures participation. Uses 12h EMA50 for stronger trend filter vs 1d, targeting 20-50 trades/year on 4h.
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
    
    # Get 12h data for EMA50 trend and Camarilla pivots (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h
    close_12h = pd.Series(df_12h['close'])
    ema_50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels on 12h (H3, L3)
    # Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_arr = df_12h['close'].values
    camarilla_h3 = close_12h_arr + 1.1 * (high_12h - low_12h) / 2
    camarilla_l3 = close_12h_arr - 1.1 * (high_12h - low_12h) / 2
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    
    # Calculate ATR(14) for stoploss on 4h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.nanmean(tr[i-13:i+1])
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA50, ATR, volume MA
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50_val = ema_50_12h_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        h3_level = camarilla_h3_aligned[i]
        l3_level = camarilla_l3_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for entry signals
            # Long: price > H3, above 12h EMA50, volume confirmation
            long_entry = (curr_close > h3_level) and (curr_close > ema_50_val) and volume_confirm
            # Short: price < L3, below 12h EMA50, volume confirmation
            short_entry = (curr_close < l3_level) and (curr_close < ema_50_val) and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Stoploss: 2 * ATR below entry
            stop_loss = entry_price - 2.0 * atr_val
            # Exit: stoploss hit OR price crosses below 12h EMA50
            if curr_low <= stop_loss or curr_close < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * atr_val
            # Exit: stoploss hit OR price crosses above 12h EMA50
            if curr_high >= stop_loss or curr_close > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_12hEMA50_Trend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 04:39
