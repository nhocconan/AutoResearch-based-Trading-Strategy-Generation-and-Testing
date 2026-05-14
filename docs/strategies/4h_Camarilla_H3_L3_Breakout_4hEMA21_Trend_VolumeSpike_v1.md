# Strategy: 4h_Camarilla_H3_L3_Breakout_4hEMA21_Trend_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.381 | +34.6% | -6.9% | 362 | PASS |
| ETHUSDT | 0.097 | +24.4% | -11.0% | 331 | PASS |
| SOLUSDT | 0.001 | +18.3% | -21.1% | 269 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.305 | -9.1% | -10.5% | 146 | FAIL |
| ETHUSDT | 0.746 | +15.2% | -5.2% | 127 | PASS |
| SOLUSDT | 0.537 | +12.2% | -6.5% | 102 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout + 4h EMA21 Trend + Volume Spike
Hypothesis: Camarilla H3/L3 levels from 4h chart breakouts with volume confirmation,
4h EMA21 trend filter for intermediate trend alignment. Targets 19-50 trades/year
to minimize fee drag while capturing momentum in both bull and bear markets.
Works in bull markets via breakout continuation and in bear markets via
mean-reversion from extreme levels when trend aligns.
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
    
    # Get 4h data for Camarilla pivots and EMA21 trend (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (H3, L3) from 4h OHLC
    # Camarilla: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    daily_high = df_4h['high'].values
    daily_low = df_4h['low'].values
    daily_close = df_4h['close'].values
    camarilla_h3 = daily_close + 1.1 * (daily_high - daily_low) / 4
    camarilla_l3 = daily_close - 1.1 * (daily_high - daily_low) / 4
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # Calculate 4h EMA21 for trend filter
    ema_21_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Calculate ATR(14) for stoploss
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma_20 = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA21_4h, ATR, and volume MA to propagate
    start_idx = max(21, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_21_4h_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema21_4h = ema_21_4h_aligned[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long: price breaks above H3 AND uptrend (price > 4h EMA21) AND volume spike
            long_condition = (curr_close > h3) and (curr_close > ema21_4h) and volume_spike
            # Short: price breaks below L3 AND downtrend (price < 4h EMA21) AND volume spike
            short_condition = (curr_close < l3) and (curr_close < ema21_4h) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price breaks below L3 (reversal signal)
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price breaks above H3 (reversal signal)
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3_L3_Breakout_4hEMA21_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 02:17
