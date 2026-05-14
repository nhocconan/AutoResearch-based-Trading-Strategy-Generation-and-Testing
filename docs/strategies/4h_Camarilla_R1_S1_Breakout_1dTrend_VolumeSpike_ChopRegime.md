# Strategy: 4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ChopRegime

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.178 | +28.9% | -18.1% | 96 | PASS |
| ETHUSDT | -0.745 | -20.7% | -35.0% | 106 | FAIL |
| SOLUSDT | 0.534 | +76.7% | -24.0% | 83 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.183 | +7.8% | -6.5% | 32 | PASS |
| SOLUSDT | 0.128 | +7.3% | -15.2% | 31 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ChopRegime
Hypothesis: Camarilla R1/S1 breakout on 4h with 1d EMA50 trend filter, volume spike confirmation, and choppiness regime filter to avoid whipsaw in sideways markets.
Only trade breakouts aligned with 1d EMA50 direction during volume expansion (>1.8x average volume) when market is trending (CHOP < 50).
Designed to work in both bull (trend-following breakouts) and bear (mean-reversion at extremes) regimes by filtering for trending conditions only.
Uses discrete sizing (0.25) and ATR-based stoploss (signal→0 when price moves against position by 2.0*ATR).
Targets 15-30 trades/year on 4h timeframe to minimize fee drag while capturing strong momentum moves in trending markets.
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
    
    # Get 4h data for Camarilla levels and ATR - primary timeframe
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for R1 and S1
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    camarilla_range = high_4h - low_4h
    r1_4h = close_4h + camarilla_range * 1.1 / 12
    s1_4h = close_4h - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Get 1d data for EMA50 trend filter - HTF
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA(50) on 1d
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate ATR(14) for stoploss on 4h
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_4h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index on 1d for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (log(n) * (max(high) - min(low)))) / log10(n)
    # Simplified: CHOP = 100 * log10(atr_sum / (log10(n) * (hh - ll))) / log10(n)
    # We'll use a rolling window of 14 periods
    tr_1d1 = high_1d[1:] - low_1d[1:]
    tr_1d2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr_1d3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr_1d1, np.maximum(tr_1d2, tr_1d3))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate sum of ATR over 14 periods
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Calculate highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Calculate Choppiness Index
    # Avoid division by zero and log of zero/negative
    denominator = np.log10(14) * (hh_1d - ll_1d)
    chop_raw = 100 * np.log10(atr_sum / denominator) / np.log10(14)
    # Replace infinities and NaNs with 50 (neutral)
    chop_raw = np.where((denominator <= 0) | np.isnan(atr_sum) | np.isnan(denominator), 50.0, chop_raw)
    chop_1d = chop_raw
    
    # Align Choppiness Index to 4h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 50, 20, 14, 14)  # Camarilla needs 2, EMA needs 50, vol needs 20, ATR needs 14, CHOP needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or 
            np.isnan(atr_4h_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(chop_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_1d_aligned[i]
        atr_val = atr_4h_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        chop_val = chop_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 1.8x 20-period average (stricter to reduce trades)
        volume_spike = vol_val > 1.8 * vol_ma_val
        
        # Choppiness regime filter: only trade when market is trending (CHOP < 50)
        trending_regime = chop_val < 50
        
        if position == 0:
            # Look for entry signals: Camarilla breakout with trend, volume, and regime confirmation
            # Long: price breaks above R1, above 1d EMA50, with volume spike, in trending regime
            long_signal = (high_val > r1_val) and (close_val > ema_val) and volume_spike and trending_regime
            # Short: price breaks below S1, below 1d EMA50, with volume spike, in trending regime
            short_signal = (low_val < s1_val) and (close_val < ema_val) and volume_spike and trending_regime
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.0*ATR
            if close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: price closes below 1d EMA50
            elif close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 3. Regime change: market becomes choppy (CHOP >= 50)
            elif chop_val >= 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.0*ATR
            if close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: price closes above 1d EMA50
            elif close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 3. Regime change: market becomes choppy (CHOP >= 50)
            elif chop_val >= 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 23:00
