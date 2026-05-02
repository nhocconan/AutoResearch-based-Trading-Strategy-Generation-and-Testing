#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: bull_power = high - EMA13, bear_power = low - EMA13
# ADX > 25 indicates trending market: only take Elder Ray signals in direction of trend
# Volume spike (2.0x 20-period average) confirms institutional participation
# Works in bull markets via bull_power > 0 + uptrend, in bear markets via bear_power < 0 + downtrend
# Targets 12-37 trades/year (50-150 total over 4 years) for 6h timeframe

name = "6h_ElderRay_1dADXRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for regime filter
    # ADX requires +DI, -DI, and TR
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = pd.Series(df_1d['low']).diff().abs()
    tr3 = abs(pd.Series(df_1d['high']) - pd.Series(df_1d['low']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d EMA(34) for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA13 and volume MA)
    start_idx = 20  # buffer for 20-period calculations
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(adx_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Determine regime: ADX > 25 = trending market
        is_trending = adx_aligned[i] > 25
        # Determine trend direction: close > EMA34 = uptrend, close < EMA34 = downtrend
        is_uptrend = close[i] > ema_34_1d_aligned[i]
        is_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: bull_power > 0 + uptrend + volume spike
            if (bull_power[i] > 0 and is_uptrend and is_trending and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: bear_power < 0 + downtrend + volume spike
            elif (bear_power[i] < 0 and is_downtrend and is_trending and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: bull_power <= 0 or trend breaks to downtrend
            if bull_power[i] <= 0 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: bear_power >= 0 or trend breaks to uptrend
            if bear_power[i] >= 0 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals