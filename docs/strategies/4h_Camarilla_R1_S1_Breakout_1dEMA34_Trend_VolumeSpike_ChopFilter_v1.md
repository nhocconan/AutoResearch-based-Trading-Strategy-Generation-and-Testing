# Strategy: 4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.283 | +30.7% | -10.6% | 171 | PASS |
| ETHUSDT | 0.013 | +21.0% | -7.9% | 164 | PASS |
| SOLUSDT | 0.478 | +53.6% | -14.4% | 149 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.392 | -3.4% | -6.7% | 73 | FAIL |
| ETHUSDT | 0.431 | +10.9% | -5.5% | 67 | PASS |
| SOLUSDT | 0.086 | +6.7% | -8.3% | 52 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h Camarilla R1/S1 Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Camarilla pivot levels (R1/S1) from daily chart act as intraday support/resistance.
Breakouts above R1 or below S1 with volume confirmation and 1d EMA34 trend filter capture
strong momentum moves. Chop filter avoids whipsaws in ranging markets. Discrete sizing (0.25)
controls drawdown. Target: 20-50 trades/year on 4h.
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
    
    # Get 1d data for Camarilla pivots and EMA34 (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels (R1, S1) from 1d OHLC
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    camarilla_r1 = daily_close + 1.1 * (daily_high - daily_low) / 12
    camarilla_s1 = daily_close - 1.1 * (daily_high - daily_low) / 12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate ATR(14) for stoploss and volatility filter
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
    
    # Calculate Choppiness Index (CHOP) for regime filter
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    if len(close) >= 14:
        atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum()
        hh = pd.Series(high).rolling(window=14, min_periods=14).max()
        ll = pd.Series(low).rolling(window=14, min_periods=14).min()
        chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
        chop_values = chop.values
    else:
        chop_values = np.full(n, 50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA34_1d, ATR, and volume MA to propagate
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(chop_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema34_1d = ema_34_1d_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        chop = chop_values[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        # Chop filter: only trade when trending (CHOP < 38.2)
        trending_regime = chop < 38.2
        
        if position == 0:
            # Long: price breaks above R1 AND uptrend (price > 1d EMA34) AND volume spike AND trending regime
            long_condition = (curr_close > r1) and (curr_close > ema34_1d) and volume_spike and trending_regime
            # Short: price breaks below S1 AND downtrend (price < 1d EMA34) AND volume spike AND trending regime
            short_condition = (curr_close < s1) and (curr_close < ema34_1d) and volume_spike and trending_regime
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price breaks below S1 (reversal signal)
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price breaks above R1 (reversal signal)
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 02:09
