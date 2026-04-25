#!/usr/bin/env python3
"""
6h Elder Ray Index with 1d ADX Regime Filter and Volume Confirmation
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear strength.
Combined with 1d ADX trend filter (ADX>25 = trending, ADX<20 = ranging) to avoid false signals in chop.
Volume spike confirms institutional participation. Works in bull (long on Bull Power >0) and bear (short on Bear Power <0).
Target: 50-150 total trades over 4 years = 12-37/year by requiring confluence of Elder Ray signal + ADX regime + volume.
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
    
    # Load 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d ADX calculation (standard 14-period)
    period = 14
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = pd.Series(df_1d['high']).sub(pd.Series(df_1d['low']))
    tr2 = pd.Series(df_1d['high']).sub(pd.Series(df_1d['close']).shift(1)).abs()
    tr3 = pd.Series(df_1d['low']).sub(pd.Series(df_1d['close']).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    adx_values = adx.values
    
    # 1d ADX aligned to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Elder Ray components on 6h timeframe
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - low   # Bear Power = EMA13 - Low
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA13 and ADX
    start_idx = max(13, 50)  # EMA13 lookback, ADX calculation
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        adx_val = adx_1d_aligned[i]
        
        # Regime filter: ADX > 25 = trending (favor Elder Ray signals), ADX < 20 = ranging (avoid)
        trending_regime = adx_val > 25
        ranging_regime = adx_val < 20
        
        if position == 0:
            # Look for entry signals - require Elder Ray signal + trending regime + volume spike
            # Long: Bull Power > 0 (bulls in control) AND trending regime AND volume spike
            long_entry = (bull_power[i] > 0) and trending_regime and vol_spike
            # Short: Bear Power > 0 (bears in control) AND trending regime AND volume spike
            short_entry = (bear_power[i] > 0) and trending_regime and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Bull Power <= 0 (bulls lose control) OR ADX drops below 20 (trend weakening)
            if (bull_power[i] <= 0) or (adx_val < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Bear Power <= 0 (bears lose control) OR ADX drops below 20 (trend weakening)
            if (bear_power[i] <= 0) or (adx_val < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ADX_Regime_VolumeSpike"
timeframe = "6h"
leverage = 1.0