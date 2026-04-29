#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R extreme reversal with volume confirmation and 1d ADX regime filter
# Williams %R(14) < -80 = oversold (long), > -20 = overbought (short) on 12h timeframe
# Volume > 1.5x 20-period average confirms reversal strength
# 1d ADX > 25 ensures we trade only in trending markets (avoid chop)
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Works in bull/bear: ADX filter avoids false signals in ranging markets, volume confirms momentum

name = "12h_WilliamsR_Extreme_VolumeSpike_1dADX_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R on 12h data
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # 1d ADX for trend filter (avoid ranging markets)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on daily data
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff().copy()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr1 = pd.Series(df_1d['high']).diff()
    tr2 = pd.Series(df_1d['low']).diff().abs()
    tr3 = pd.Series(df_1d['close']).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    smoothed_plus_dm = plus_dm.ewm(alpha=1/14, adjust=False).mean()
    smoothed_minus_dm = minus_dm.abs().ewm(alpha=1/14, adjust=False).mean()
    smoothed_tr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    plus_di = 100 * (smoothed_plus_dm / smoothed_tr.replace(0, np.nan))
    minus_di = 100 * (smoothed_minus_dm / smoothed_tr.replace(0, np.nan))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period, 20, 30)  # warmup for Williams %R, volume MA, and ADX
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if np.isnan(williams_r[i]) or np.isnan(vol_ma_20[i]) or np.isnan(adx_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_williams_r = williams_r[i]
        curr_volume_confirm = volume_confirm[i]
        curr_adx = adx_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade in trending markets (ADX > 25) with volume confirmation
            if curr_adx > 25 and curr_volume_confirm:
                # Bullish entry: Williams %R oversold (< -80) reversal
                if curr_williams_r < -80:
                    signals[i] = 0.30
                    position = 1
                # Bearish entry: Williams %R overbought (> -20) reversal
                elif curr_williams_r > -20:
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when Williams %R returns from oversold (> -50) or overbought condition
            if curr_williams_r > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit when Williams %R returns from overbought (< -50) or oversold condition
            if curr_williams_r < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals