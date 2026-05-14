# Strategy: 4h_Camarilla_H4L4_Breakout_1dEMA34_ADXFilter_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.564 | +40.4% | -5.5% | 221 | PASS |
| ETHUSDT | 0.716 | +54.0% | -7.5% | 243 | PASS |
| SOLUSDT | 0.245 | +34.3% | -18.5% | 184 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.766 | -5.5% | -6.2% | 100 | FAIL |
| ETHUSDT | 1.085 | +19.3% | -5.8% | 89 | PASS |
| SOLUSDT | 1.004 | +18.4% | -5.1% | 70 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_H4L4_Breakout_1dEMA34_ADXFilter_VolumeSpike
Hypothesis: Camarilla H4/L4 breakout with 1d EMA34 trend filter, ADX(14) regime filter for trending markets, and volume spike confirmation.
Uses ATR-based trailing stoploss to reduce whipsaw. ADX > 25 ensures we only trade in trending conditions, improving bear market performance.
H4/L4 levels are stronger support/resistance than H3/L3, reducing false breakouts.
1d EMA34 provides reliable long-term trend filter. Volume spike confirms institutional participation.
Designed for 19-50 trades/year (75-200 over 4 years) to minimize fee drag.
Works in bull markets via breakout continuation and bear markets via trend following with tight stops and regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla calculation and EMA34 (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 trend filter (loaded ONCE)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Prior 1d bar OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: H4, L4 (stronger intraday support/resistance than H3/L3)
    camarilla_range = prev_high - prev_low
    h4 = prev_close + camarilla_range * 1.1 / 2
    l4 = prev_close - camarilla_range * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ADX(14) for regime filter - only trade when ADX > 25 (trending market)
    # Calculate using 1d data for stability, then align to 4h
    plus_dm = np.where((df_1d['high'].diff()) > (df_1d['low'].diff().abs()), df_1d['high'].diff(), 0)
    minus_dm = np.where((df_1d['low'].diff().abs()) > (df_1d['high'].diff()), df_1d['low'].diff().abs(), 0)
    plus_dm = np.where(plus_dm < 0, 0, plus_dm)
    minus_dm = np.where(minus_dm < 0, 0, minus_dm)
    
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    plus_di_1d = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_1d)
    minus_di_1d = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_1d)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_1d_values = adx_1d.fillna(0).values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_values)
    
    # ATR for volatility-based stoploss (14-period) on 4h data
    tr1_4h = high[1:] - low[1:]
    tr2_4h = np.abs(high[1:] - close[:-1])
    tr3_4h = np.abs(low[1:] - close[:-1])
    tr_4h = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for 1d EMA (34), 1d ADX (14+14), volume MA (20), ATR (14)
    start_idx = max(34, 28, 20, 14)  # EMA34 + ADX(14+14) + vol MA20 + ATR14
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_4h[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla H4/L4 breakout + volume spike + 1d EMA34 trend alignment + ADX > 25 (trending)
            long_breakout = curr_high > h4_aligned[i]
            short_breakout = curr_low < l4_aligned[i]
            
            # Trend filter: price must be on correct side of 1d EMA34
            long_trend = curr_close > ema_34_1d_aligned[i]
            short_trend = curr_close < ema_34_1d_aligned[i]
            
            # Regime filter: only trade when ADX > 25 (trending market)
            trending = adx_1d_aligned[i] > 25
            
            long_entry = (long_breakout and volume_spike[i] and long_trend and trending)
            short_entry = (short_breakout and volume_spike[i] and short_trend and trending)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: track highest price for trailing stop
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit when price closes below Camarilla H4 (failed breakout) or trend reverses or ATR stop hit or ADX drops (losing trend)
            atr_stop = highest_since_entry - 2.5 * atr_4h[i]
            if curr_close < h4_aligned[i] or curr_close < ema_34_1d_aligned[i] or curr_close < atr_stop or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: track lowest price for trailing stop
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit when price closes above Camarilla L4 (failed breakout) or trend reverses or ATR stop hit or ADX drops (losing trend)
            atr_stop = lowest_since_entry + 2.5 * atr_4h[i]
            if curr_close > l4_aligned[i] or curr_close > ema_34_1d_aligned[i] or curr_close > atr_stop or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H4L4_Breakout_1dEMA34_ADXFilter_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 09:10
