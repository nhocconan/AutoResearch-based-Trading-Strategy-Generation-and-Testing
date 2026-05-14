# Strategy: 4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_Regime_ADX

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.415 | +43.4% | -10.5% | 238 | PASS |
| ETHUSDT | 0.130 | +26.3% | -16.0% | 223 | PASS |
| SOLUSDT | 0.667 | +100.5% | -22.0% | 210 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.045 | -5.4% | -9.0% | 92 | FAIL |
| ETHUSDT | 0.668 | +17.9% | -12.9% | 75 | PASS |
| SOLUSDT | 0.424 | +13.2% | -11.5% | 69 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_Regime_ADX
Hypothesis: Camarilla H3/L3 breakout on 4h with 1d EMA34 trend filter, volume confirmation, and ADX regime filter.
Works in bull markets (breakouts with trend) and bear markets (fades from extremes with volume).
ADX > 25 ensures we only trade in trending markets, reducing whipsaws in ranging conditions.
Uses discrete position sizing (0.30) to limit fee drag. Targets 20-40 trades/year.
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
    
    # Get 1d data for EMA34 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Camarilla levels: H3/L3
    camarilla_h3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_l3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align to 4h timeframe (completed 1d bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # ADX regime filter: only trade when ADX > 25 (trending market)
    # Calculate ADX using 14-period
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smoothed values
        atr = np.zeros(len(high))
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros(len(high))
        minus_di = np.zeros(len(high))
        dx = np.zeros(len(high))
        
        for i in range(period, len(high)):
            if atr[i] > 0:
                plus_di[i] = 100 * (plus_dm[i] / atr[i])
                minus_di[i] = 100 * (minus_dm[i] / atr[i])
                if (plus_di[i] + minus_di[i]) > 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros(len(high))
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_values = calculate_adx(high, low, close, 14)
    adx_filter = adx_values > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Camarilla (1 bar), EMA34 (34), volume MA (20), ADX (28)
    start_idx = max(1, 34, 20, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(adx_values[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price closes above H3 + 1d uptrend + volume spike + ADX > 25
            long_setup = (close[i] > camarilla_h3_aligned[i]) and \
                         (close[i] > ema_34_1d_aligned[i]) and \
                         volume_spike[i] and \
                         adx_filter[i]
            # Short: price closes below L3 + 1d downtrend + volume spike + ADX > 25
            short_setup = (close[i] < camarilla_l3_aligned[i]) and \
                          (close[i] < ema_34_1d_aligned[i]) and \
                          volume_spike[i] and \
                          adx_filter[i]
            
            if long_setup:
                signals[i] = 0.30
                position = 1
            elif short_setup:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit: price closes below L3 OR 1d trend turns down
            if (close[i] < camarilla_l3_aligned[i]) or \
               (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit: price closes above H3 OR 1d trend turns up
            if (close[i] > camarilla_h3_aligned[i]) or \
               (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_Regime_ADX"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 12:52
