# Strategy: 6h_Camarilla_H3L3_Breakout_12hEMA34_Trend_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.244 | +33.1% | -11.6% | 66 | PASS |
| ETHUSDT | 0.017 | +18.5% | -13.4% | 55 | PASS |
| SOLUSDT | 0.939 | +149.6% | -21.8% | 52 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -3.121 | -6.1% | -6.2% | 4 | FAIL |
| ETHUSDT | 1.029 | +24.0% | -8.0% | 20 | PASS |
| SOLUSDT | -1.175 | -10.9% | -15.3% | 24 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 12h EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h EMA34 for trend filter (price > EMA34 = uptrend, price < EMA34 = downtrend).
- Entry: Long when close breaks above Camarilla H3 level AND price > 12h EMA34 AND volume > 2.0 * 6h volume MA(20);
         Short when close breaks below Camarilla L3 level AND price < 12h EMA34 AND volume > 2.0 * 6h volume MA(20).
- Exit: Close below/above Camarilla L3/H3 levels for profit-taking, with ATR-based stoploss (2.0 * ATR(14)).
- Signal size: 0.25 discrete to control fee drag.
- Uses Camarilla pivot levels from 1d data for structure, volume confirmation for participation,
  12h EMA34 trend filter to avoid counter-trend trades, and ATR for risk management.
- Designed to work in both bull and bear markets via trend filter and tight entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for volume MA(20) and ATR(14)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate ATR(14) for 6h timeframe
    tr1 = high_6h[1:] - low_6h[1:]
    tr2 = np.abs(high_6h[1:] - close_6h[:-1])
    tr3 = np.abs(low_6h[1:] - close_6h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high_6h[0] - low_6h[0]], tr])  # first TR is high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) for 6h timeframe
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Camarilla pivot levels (H3, L3, H4, L4)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from prior 1d OHLC
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low)
    #           L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
    camarilla_h4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d)
    camarilla_l4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align 1d Camarilla levels to 6h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34 = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA34 to 6h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # EMA34 needs 34, volume MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vol_ma_6h[i]) or 
            np.isnan(atr14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_atr = atr14[i]
        
        # Volume confirmation: 2.0x threshold for strict entry
        vol_confirm = curr_volume > 2.0 * vol_ma_6h[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Close breaks above Camarilla H3 AND price > 12h EMA34 (uptrend)
                if curr_close > camarilla_h3_aligned[i] and curr_close > ema_34_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Short: Close breaks below Camarilla L3 AND price < 12h EMA34 (downtrend)
                elif curr_close < camarilla_l3_aligned[i] and curr_close < ema_34_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        elif position == 1:
            # Long position: check exit conditions
            # Stoploss: 2.0 * ATR below entry
            stoploss = entry_price - 2.0 * curr_atr
            # Profit take: close below Camarilla L3
            if curr_close < stoploss or curr_close < camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: check exit conditions
            # Stoploss: 2.0 * ATR above entry
            stoploss = entry_price + 2.0 * curr_atr
            # Profit take: close above Camarilla H3
            if curr_close > stoploss or curr_close > camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_12hEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-24 17:59
