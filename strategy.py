#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX Regime + Volume Spike
# Elder Ray measures bull/bear power: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# In ranging markets (ADX<25): fade extremes - long when Bear Power < -0.5*ATR and volume spike, short when Bull Power > 0.5*ATR and volume spike
# In trending markets (ADX>=25): only trade with trend - long when Bull Power > 0 and ADX rising, short when Bear Power < 0 and ADX falling
# Volume spike (>1.5x 20-period EMA) confirms momentum. Designed for 6h timeframe targeting 50-150 total trades.
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown in both bull and bear markets.

name = "6h_ElderRay_1dADX_Regime_Volume"
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
    
    # Get 1d data for Elder Ray and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(13) for Elder Ray
    ema_13 = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d Elder Ray components
    bull_power = df_1d['high'].values - ema_13
    bear_power = df_1d['low'].values - ema_13
    
    # Calculate 1d ATR(14) for Elder Ray thresholds
    tr1 = pd.Series(df_1d['high']).sub(df_1d['low'])
    tr2 = pd.Series(df_1d['high']).sub(df_1d['close'].shift(1)).abs()
    tr3 = pd.Series(df_1d['low']).sub(df_1d['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ADX(14)
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr_atr = tr.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / tr_atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / tr_atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Determine regime: ranging (ADX<25) or trending (ADX>=25)
            if adx_aligned[i] < 25:
                # Ranging market: mean reversion at extremes
                if bear_power_aligned[i] < (-0.5 * atr_aligned[i]) and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                elif bull_power_aligned[i] > (0.5 * atr_aligned[i]) and volume_confirm:
                    signals[i] = -0.25
                    position = -1
            else:
                # Trending market: only trade with trend
                # ADX rising: current ADX > previous ADX
                adx_rising = i > 100 and adx_aligned[i] > adx_aligned[i-1]
                adx_falling = i > 100 and adx_aligned[i] < adx_aligned[i-1]
                
                # Long: Bull Power > 0 (bulls in control) + volume + ADX rising
                if (bull_power_aligned[i] > 0 and 
                    volume_confirm and 
                    adx_rising):
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power < 0 (bears in control) + volume + ADX falling
                elif (bear_power_aligned[i] < 0 and 
                      volume_confirm and 
                      adx_falling):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Bear Power > -0.2*ATR (weakening bears) OR ADX < 20 OR volume drops
            if (bear_power_aligned[i] > (-0.2 * atr_aligned[i]) or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power < 0.2*ATR (weakening bulls) OR ADX < 20 OR volume drops
            if (bull_power_aligned[i] < (0.2 * atr_aligned[i]) or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals