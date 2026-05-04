#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) + 1w ADX regime + volume confirmation
# In strong trends (1w ADX>=25), we trade with the trend: long when Bull Power > 0, short when Bear Power < 0.
# In weak trends/ranging (1w ADX<25), we fade extremes: short when Bull Power > 0 + overbought, long when Bear Power < 0 + oversold.
# Volume confirmation (>1.3x 20-period EMA) reduces false signals. Designed for 6h timeframe targeting 50-150 total trades over 4 years.
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "6h_ElderRay_1wADX_Regime_Volume"
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
    
    # Get 1w data for ADX and EMA13 (for Elder Ray)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w ADX (14-period)
    plus_dm = pd.Series(df_1w['high']).diff()
    minus_dm = pd.Series(df_1w['low']).diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr1 = pd.Series(df_1w['high']).sub(df_1w['low'])
    tr2 = pd.Series(df_1w['high']).sub(df_1w['close'].shift(1)).abs()
    tr3 = pd.Series(df_1w['low']).sub(df_1w['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    # Calculate 1w EMA13 (for Elder Ray calculations)
    ema13 = pd.Series(df_1w['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align 1w indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx.values)
    ema13_aligned = align_htf_to_ltf(prices, df_1w, ema13)
    
    # Calculate Elder Ray components on 6h timeframe using aligned 1w EMA13
    bull_power = high - ema13_aligned  # Bull Power = High - EMA13
    bear_power = low - ema13_aligned   # Bear Power = Low - EMA13
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.3 x 20-period EMA
        volume_confirm = volume[i] > (1.3 * vol_ema_20[i])
        
        if position == 0:
            # Determine regime: weak trend/ranging (ADX<25) or strong trend (ADX>=25)
            if adx_aligned[i] < 25:
                # Weak trend/ranging market: fade extremes
                # Short when Bull Power > 0 (overbought) + volume confirmation
                if bull_power[i] > 0 and volume_confirm:
                    signals[i] = -0.25
                    position = -1
                # Long when Bear Power < 0 (oversold) + volume confirmation
                elif bear_power[i] < 0 and volume_confirm:
                    signals[i] = 0.25
                    position = 1
            else:
                # Strong trend market: trade with the trend
                # Long when Bull Power > 0 (bulls in control)
                if bull_power[i] > 0 and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                # Short when Bear Power < 0 (bears in control)
                elif bear_power[i] < 0 and volume_confirm:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Bull Power turns negative OR ADX weakening (<20) OR volume drops
            if (bull_power[i] <= 0 or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power turns positive OR ADX weakening (<20) OR volume drops
            if (bear_power[i] >= 0 or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals