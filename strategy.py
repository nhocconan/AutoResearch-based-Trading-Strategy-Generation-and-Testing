#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band breakout with 1d ATR regime filter and volume confirmation
# Long when price breaks above 6h BB upper (20, 2.0) + 1d ATR(14) > 1d ATR(50) + volume > 1.5x 20-period avg
# Short when price breaks below 6h BB lower (20, 2.0) + 1d ATR(14) > 1d ATR(50) + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-37/year).
# Bollinger Bands provide volatility-based breakout levels. ATR regime filter ensures we only trade
# when volatility is expanding (avoiding chop). Works in bull markets (trend continuation) and bear markets
# (strong downtrends) by requiring expanding volatility regime.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d Indicator: ATR regime filter (expanding volatility) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    close_1d_shift = np.roll(close_1d, 1)
    high_1d_shift[0] = high_1d[0]
    low_1d_shift[0] = low_1d[0]
    close_1d_shift[0] = close_1d[0]
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d_shift)
    tr3 = np.abs(low_1d - close_1d_shift)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR(14) and ATR(50) using Wilder's smoothing
    def calculate_wilder_atr(tr_vals, period):
        atr = np.zeros_like(tr_vals)
        if len(tr_vals) < period:
            return atr
        atr[period-1] = np.mean(tr_vals[:period])
        for i in range(period, len(tr_vals)):
            atr[i] = (atr[i-1] * (period-1) + tr_vals[i]) / period
        return atr
    
    atr_14 = calculate_wilder_atr(tr, 14)
    atr_50 = calculate_wilder_atr(tr, 50)
    
    # ATR regime: expanding volatility when ATR(14) > ATR(50)
    atr_regime = atr_14 > atr_50
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime.astype(float))
    
    # === 6h Indicator: Bollinger Bands (20, 2.0) ===
    bb_window = 20
    bb_std = 2.0
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=bb_window, min_periods=bb_window).mean().values
    bb_std_dev = close_s.rolling(window=bb_window, min_periods=bb_window).std().values
    bb_upper = bb_mid + (bb_std_dev * bb_std)
    bb_lower = bb_mid - (bb_std_dev * bb_std)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(bb_window, 50) + 20  # BB(20) + ATR(50) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(atr_regime_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 6h BB upper (20, 2.0)
        # 2. Volatility regime expanding (1d ATR(14) > ATR(50))
        # 3. Volume confirmation
        if (close[i] > bb_upper[i]) and \
           (atr_regime_aligned[i] > 0.5) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 6h BB lower (20, 2.0)
        # 2. Volatility regime expanding (1d ATR(14) > ATR(50))
        # 3. Volume confirmation
        elif (close[i] < bb_lower[i]) and \
             (atr_regime_aligned[i] > 0.5) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_BB20_2.0_1dATR_Regime_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0