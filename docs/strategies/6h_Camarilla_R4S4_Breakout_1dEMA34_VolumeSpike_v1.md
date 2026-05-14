# Strategy: 6h_Camarilla_R4S4_Breakout_1dEMA34_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.396 | +40.1% | -10.8% | 115 | PASS |
| ETHUSDT | 0.214 | +31.4% | -12.8% | 111 | PASS |
| SOLUSDT | 0.660 | +89.2% | -18.9% | 85 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.989 | -4.5% | -8.3% | 47 | FAIL |
| ETHUSDT | 1.291 | +29.0% | -6.4% | 33 | PASS |
| SOLUSDT | -0.277 | +0.8% | -14.2% | 32 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h Camarilla R4/S4 Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Camarilla pivot levels (R4/S4) act as strong breakout/continuation levels. 
Breakouts above R4 or below S4 with volume confirmation and 1d EMA34 trend filter capture strong momentum moves.
Works in both bull/bear markets: in bull, longs above R4 with uptrend; in bear, shorts below S4 with downtrend.
Volume spike confirms institutional participation. Discrete sizing (0.25) limits fee drag.
Target: 12-37 trades/year (50-150 over 4 years).
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
    
    # Get 1d data for EMA34 trend filter and Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels (R4, S4) from prior 1d bar
    # R4 = close + 1.1*(high-low)/2, S4 = close - 1.1*(high-low)/2
    camarilla_r4 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 2
    camarilla_s4 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 2
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4.values)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4.values)
    
    # Calculate ATR for volatility filtering (using 14 periods)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for ATR (14) and EMA34 (34)
    start_idx = max(14, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or np.isnan(atr[i]):
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
        camarilla_r4_val = camarilla_r4_aligned[i]
        camarilla_s4_val = camarilla_s4_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout conditions: price breaks above R4 or below S4
        bullish_breakout = curr_close > camarilla_r4_val
        bearish_breakout = curr_close < camarilla_s4_val
        
        # Exit conditions: reverse breakout or trend rejection
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Exit on bearish breakout below S4 or trend rejection
                if bearish_breakout or curr_close < ema_trend:
                    exit_signal = True
                    
            elif position == -1:
                # Exit on bullish breakout above R4 or trend rejection
                if bullish_breakout or curr_close > ema_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Camarilla breakout + trend alignment + volume spike
        if position == 0:
            # Long: break above R4 AND price above 1d EMA34
            long_condition = bullish_breakout and (curr_close > ema_trend) and volume_spike
            # Short: break below S4 AND price below 1d EMA34
            short_condition = bearish_breakout and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 00:58
