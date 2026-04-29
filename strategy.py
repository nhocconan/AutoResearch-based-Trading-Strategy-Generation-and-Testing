#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + Volume Spike + 1d ADX Trend Filter
# Williams %R identifies overbought/oversold conditions (>80 oversold, <-20 overbought)
# Volume spike (>2.0x 30-period average) confirms institutional participation at extremes
# 1d ADX > 25 ensures trades align with strong daily trend (avoid chop)
# Works in bull/bear: ADX filters for trending markets, Williams %R + volume spots exhaustion
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_WilliamsR_Extreme_VolumeSpike_1dADX_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Volume confirmation: volume > 2.0x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (2.0 * vol_ma_30)
    
    # 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX (14-period) on 1d data
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff().copy()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)
    
    tr1 = pd.Series(df_1d['high']) - pd.Series(df_1d['low'])
    tr2 = abs(pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1))
    tr3 = abs(pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean().values
    
    # Align daily ADX to 6h timeframe (wait for daily bar to close)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # warmup for volume MA and Williams %R
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if np.isnan(williams_r[i]) or np.isnan(vol_ma_30[i]) or np.isnan(adx_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_williams_r = williams_r[i]
        curr_volume_confirm = volume_confirm[i]
        curr_adx = adx_aligned[i]
        
        # Only trade in trending markets (ADX > 25) with volume confirmation
        if curr_adx > 25.0 and curr_volume_confirm:
            if position == 0:  # Flat - look for new entries
                # Bullish entry: Williams %R oversold (< -80) with volume
                if curr_williams_r < -80.0:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R overbought (> -20) with volume
                elif curr_williams_r > -20.0:
                    signals[i] = -0.25
                    position = -1
            
            elif position == 1:  # Long position
                # Exit when Williams %R returns to neutral (> -50) or overbought (> -20)
                if curr_williams_r > -20.0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            
            elif position == -1:  # Short position
                # Exit when Williams %R returns to neutral (< -50) or oversold (< -80)
                if curr_williams_r < -80.0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals