#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R extreme reversal with 1d ADX regime filter and volume confirmation
# In ranging markets (1d ADX < 25): Williams %R < -80 = long (oversold), > -20 = short (overbought)
# In trending markets (1d ADX >= 25): Williams %R pullback from extreme in trend direction
# Volume confirmation (>1.5x 20-period EMA) ensures participation. Uses discrete sizing (0.25) to minimize fees.
# Designed for 12h timeframe targeting 50-150 total trades over 4 years (12-37/year).
# BTC/ETH edge: Williams %R captures short-term exhaustion; ADX regime avoids whipsaws in trends; volume confirms institutional interest.

name = "12h_WilliamsR_1dADX_Regime_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX and Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr1 = pd.Series(df_1d['high']).sub(df_1d['low'])
    tr2 = pd.Series(df_1d['high']).sub(df_1d['close'].shift(1)).abs()
    tr3 = pd.Series(df_1d['low']).sub(df_1d['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    # Calculate 1d Williams %R (14-period)
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - df_1d['close']) / (highest_high - lowest_low)
    
    # Align 1d indicators to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r.values)
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Determine regime: ranging (ADX<25) or trending (ADX>=25)
            if adx_aligned[i] < 25:
                # Ranging market: mean reversion from extremes
                if williams_r_aligned[i] <= -80 and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                elif williams_r_aligned[i] >= -20 and volume_confirm:
                    signals[i] = -0.25
                    position = -1
            else:
                # Trending market: pullback from extreme in trend direction
                # Calculate 1d +DI and -DI for trend direction
                plus_di_1d = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / atr)
                minus_di_1d = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / atr)
                plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di_1d.values)
                minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di_1d.values)
                
                # Long: pullback from oversold in uptrend (+DI > -DI)
                if (williams_r_aligned[i] <= -80 and 
                    volume_confirm and 
                    plus_di_aligned[i] > minus_di_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: pullback from overbought in downtrend (-DI > +DI)
                elif (williams_r_aligned[i] >= -20 and 
                      volume_confirm and 
                      minus_di_aligned[i] > plus_di_aligned[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Williams %R returns to neutral (-50) OR ADX weakening (<20) OR volume drops
            if (williams_r_aligned[i] >= -50 or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to neutral (-50) OR ADX weakening (<20) OR volume drops
            if (williams_r_aligned[i] <= -50 or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals