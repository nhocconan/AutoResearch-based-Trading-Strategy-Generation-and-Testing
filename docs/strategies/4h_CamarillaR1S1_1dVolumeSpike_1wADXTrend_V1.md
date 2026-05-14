# Strategy: 4h_CamarillaR1S1_1dVolumeSpike_1wADXTrend_V1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.372 | +10.0% | -6.4% | 395 | FAIL |
| ETHUSDT | 0.551 | +45.2% | -6.2% | 359 | PASS |
| SOLUSDT | -0.278 | +5.2% | -16.3% | 231 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.078 | +21.1% | -11.5% | 164 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R1/S1 breakout with 1d volume spike and 1w ADX trend filter.
# Long when price breaks above Camarilla R1 (1d) AND volume > 2.0x 20-period average AND 1w ADX > 25.
# Short when price breaks below Camarilla S1 (1d) AND volume > 2.0x 20-period average AND 1w ADX > 25.
# Exit when price returns to Camarilla pivot point (PP) or ATR(10) < ATR(30) (contracting volatility).
# Uses discrete position size 0.25. Camarilla levels from 1d provide institutional reference points,
# volume confirmation reduces false breakouts, 1w ADX ensures we only trade in trending regimes.
# Target: 75-200 total trades over 4 years (19-50/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Camarilla pivot levels (R1, S1, PP) ===
    # Camarilla: PP = (H + L + C) / 3
    # R1 = C + ((H - L) * 1.1 / 12)
    # S1 = C - ((H - L) * 1.1 / 12)
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + ((high_1d - low_1d) * 1.1 / 12.0)
    s1_1d = close_1d - ((high_1d - low_1d) * 1.1 / 12.0)
    
    # Align Camarilla levels to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get 1w data once before loop for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: ADX(14) for trend filter ===
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Get 4h data for volume and ATR
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Volume moving average (20-period) on 4h
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    # True Range for ATR calculation
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(10) and ATR(30) for regime filter
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    
    # Align ATR values to 4h timeframe
    atr_10_aligned = align_htf_to_ltf(prices, df_4h, atr_10)
    atr_30_aligned = align_htf_to_ltf(prices, df_4h, atr_30)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(atr_10_aligned[i]) or np.isnan(atr_30_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        atr_10_val = atr_10_aligned[i]
        atr_30_val = atr_30_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to pivot point or ATR contracts
            if price <= pp_val or atr_10_val < atr_30_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to pivot point or ATR contracts
            if price >= pp_val or atr_10_val < atr_30_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: volume > 2.0x 20-period average (stricter to reduce trades)
            vol_filter = vol > 2.0 * vol_ma_val
            
            # Trend filter: 1w ADX > 25 (trending regime)
            trend_filter = adx_val > 25
            
            # LONG: price breaks above Camarilla R1 with volume and trend confirmation
            if price > r1_val and vol_filter and trend_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price breaks below Camarilla S1 with volume and trend confirmation
            elif price < s1_val and vol_filter and trend_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_CamarillaR1S1_1dVolumeSpike_1wADXTrend_V1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-16 06:09
