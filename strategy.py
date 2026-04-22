#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + ADX trend filter with volume confirmation
# Long when Williams %R < -80 (oversold) + ADX > 25 (trending) + volume > 1.5x 20-period average
# Short when Williams %R > -20 (overbought) + ADX > 25 (trending) + volume spike
# Exit when Williams %R returns to neutral range (-50 to -50) or ADX < 20
# Designed for 6h timeframe with ~15-25 trades/year to minimize fee drag.
# Williams %R identifies extremes, ADX ensures we trade with trend, volume confirms conviction.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 6h data for Williams %R and ADX calculation (once before loop)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], -50).fillna(-50).values
    
    # Calculate ADX (14-period)
    tr1 = pd.Series(high_6h - low_6h)
    tr2 = pd.Series(abs(high_6h - close_6h.shift()))
    tr3 = pd.Series(abs(low_6h - close_6h.shift()))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    dm_plus = pd.Series(high_6h - high_6h.shift()).clip(lower=0)
    dm_minus = pd.Series(low_6h.shift() - low_6h).clip(lower=0)
    
    # Smooth DM and TR
    dm_plus_smooth = dm_plus.rolling(window=14, min_periods=14).sum()
    dm_minus_smooth = dm_minus.rolling(window=14, min_periods=14).sum()
    tr_smooth = tr.rolling(window=14, min_periods=14).sum()
    
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    williams_r = williams_r.values
    adx = adx.values
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(adx[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        williams_val = williams_r[i]
        adx_val = adx[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) + ADX > 25 (trending) + volume spike
            if williams_val < -80 and adx_val > 25 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) + ADX > 25 (trending) + volume spike
            elif williams_val > -20 and adx_val > 25 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Williams %R returns to neutral range (-50 to -50) or ADX < 20
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R > -50 or ADX < 20
                if williams_val > -50 or adx_val < 20:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R < -50 or ADX < 20
                if williams_val < -50 or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_ADX25_Volume"
timeframe = "6h"
leverage = 1.0