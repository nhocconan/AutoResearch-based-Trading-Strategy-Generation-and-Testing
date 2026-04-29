#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions: > -20 = overbought, < -80 = oversold
# In ranging markets (ADX < 25), mean reversion at extremes works well
# 1d EMA34 ensures we trade with higher timeframe trend to avoid fighting the trend
# Volume confirmation (>1.5x 20-period average) filters for institutional participation
# Discrete sizing (0.25) minimizes fee churn
# Effective in both bull and bear markets: mean reversion in ranges, trend following when aligned
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe

name = "6h_WilliamsR_MeanRev_1dEMA34_VolumeConfirm_v1"
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
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R on 6h timeframe: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # Calculate ADX for regime detection (trend vs range)
    # ADX < 25 indicates ranging market suitable for mean reversion
    plus_dm = pd.Series(high).diff()
    minus_dm = pd.Series(low).diff().copy()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr1 = pd.Series(high).diff().abs()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Calculate 20-period average volume for confirmation (on 6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 14, 20)  # 1d EMA34, Williams %R, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_williams_r = williams_r[i]
        curr_adx = adx[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Regime filter: only mean revert in ranging markets (ADX < 25)
        ranging_market = curr_adx < 25
        
        # Williams %R conditions for mean reversion
        oversold = curr_williams_r < -80  # Oversold condition
        overbought = curr_williams_r > -20  # Overbought condition
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Williams %R returns from oversold OR trend strengthens (ADX >= 25) OR price crosses 1d EMA34
            if (curr_williams_r > -50) or (curr_adx >= 25) or (curr_close < curr_ema_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R returns from overbought OR trend strengthens (ADX >= 25) OR price crosses 1d EMA34
            if (curr_williams_r < -50) or (curr_adx >= 25) or (curr_close > curr_ema_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Williams %R oversold AND ranging market AND above 1d EMA34 AND volume confirmation
            if (oversold and 
                ranging_market and 
                curr_close > curr_ema_1d and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought AND ranging market AND below 1d EMA34 AND volume confirmation
            elif (overbought and 
                  ranging_market and 
                  curr_close < curr_ema_1d and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals