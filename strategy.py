#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d ATR regime filter and volume spike confirmation.
- Long when price breaks above Camarilla H3 AND 1d ATR(14) > 1.5 * 1d ATR(50) (high volatility regime) AND volume > 2.0 * 20-period average volume
- Short when price breaks below Camarilla L3 AND 1d ATR(14) > 1.5 * 1d ATR(50) (high volatility regime) AND volume > 2.0 * 20-period average volume
- Exit on opposite Camarilla level (L3 for long exit, H3 for short exit)
- Uses 4h primary with 1d HTF to target 75-200 trades over 4 years (19-50/year)
- Camarilla provides precise intraday support/resistance; ATR regime ensures trades only in high volatility environments; volume spike confirms momentum
- Designed to work in both bull (breakouts with momentum) and bear (breakouts with momentum) markets
- Signal size: 0.25 discrete levels to minimize fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels (H3, L3) on 4h data using previous day's OHLC
    # Camarilla: H3 = close + 1.1 * (high - low) / 2, L3 = close - 1.1 * (high - low) / 2
    # Using previous bar's OHLC to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan  # First bar has no previous
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Calculate 1d ATR(14) and ATR(50) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate True Range for 1d data
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # True Range: max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_daily_close = np.roll(daily_close, 1)
    prev_daily_close[0] = np.nan
    
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - prev_daily_close)
    tr3 = np.abs(daily_low - prev_daily_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) and ATR(50)
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Regime: high volatility when ATR(14) > 1.5 * ATR(50)
    high_vol_regime = atr_14 > (1.5 * atr_50)
    high_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, high_vol_regime)
    
    # Volume confirmation: volume > 2.0 * 20-period average (strong spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20) + 1  # Need Camarilla (uses prev bar), ATR50, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(high_vol_regime_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Camarilla H3 AND high volatility regime AND volume confirmation
            if close[i] > camarilla_h3[i] and high_vol_regime_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla L3 AND high volatility regime AND volume confirmation
            elif close[i] < camarilla_l3[i] and high_vol_regime_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below Camarilla L3 (opposite level)
            if close[i] < camarilla_l3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above Camarilla H3 (opposite level)
            if close[i] > camarilla_h3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_1dATRRegime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0