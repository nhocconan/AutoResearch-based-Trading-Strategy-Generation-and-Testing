#!/usr/bin/env python3
"""
1h Volume Spike + 4h EMA Trend + Chop Regime Filter
Hypothesis: On 1h timeframe, enter long when price breaks above recent swing high with volume spike (>2x avg) 
and 4h EMA50 confirms uptrend; enter short when price breaks below recent swing low with volume spike 
and 4h EMA50 confirms downtrend. Use chop regime filter (CHOP<38.2 on 1h) to only trade in trending markets.
Uses discrete sizing (0.20) and session filter (08-20 UTC) to reduce noise. Target: 15-35 trades/year.
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
    
    # Get 4h data for EMA50 trend filter (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate ATR(14) for stoploss and swing points
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Calculate swing high/low (10-period) for breakout levels
    swing_high = pd.Series(high).rolling(window=10, min_periods=10).max().values
    swing_low = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma_20 = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    # Calculate Choppiness Index (CHOP) for regime filter (14-period)
    if len(close) >= 14:
        atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum()
        hh = pd.Series(high).rolling(window=14, min_periods=14).max()
        ll = pd.Series(low).rolling(window=14, min_periods=14).min()
        chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
        chop_values = chop.values
    else:
        chop_values = np.full(n, 50.0)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA50_4h, ATR, swing points, volume MA to propagate
    start_idx = max(50, 14, 10)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(swing_high[i]) or 
            np.isnan(swing_low[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(chop_values[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema50_4h = ema_50_4h_aligned[i]
        swing_high_val = swing_high[i]
        swing_low_val = swing_low[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        chop = chop_values[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        # Chop filter: only trade when trending (CHOP < 38.2)
        trending_regime = chop < 38.2
        
        if position == 0:
            # Long: price breaks above swing high AND uptrend (price > 4h EMA50) AND volume spike AND trending regime
            long_condition = (curr_close > swing_high_val) and (curr_close > ema50_4h) and volume_spike and trending_regime
            # Short: price breaks below swing low AND downtrend (price < 4h EMA50) AND volume spike AND trending regime
            short_condition = (curr_close < swing_low_val) and (curr_close < ema50_4h) and volume_spike and trending_regime
            
            if long_condition:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price breaks below swing low (reversal signal)
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < swing_low_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price breaks above swing high (reversal signal)
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > swing_high_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_VolumeSpike_4hEMA50_Trend_SwingBreakout_ChopFilter_v1"
timeframe = "1h"
leverage = 1.0