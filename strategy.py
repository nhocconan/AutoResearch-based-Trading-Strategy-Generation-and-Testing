#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1d ADX regime filter and volume confirmation
# In ranging markets (1d ADX < 20): fade extreme Elder Ray readings (Bull Power < -std AND Bear Power > std for mean reversion)
# In trending markets (1d ADX >= 20): follow Elder Ray momentum (Bull Power > 0 for long, Bear Power < 0 for short)
# Volume confirmation (>1.3x 20-period EMA) ensures institutional participation. Uses discrete sizing (0.25) to minimize fees.
# Designed for 6h timeframe targeting 50-150 total trades over 4 years (12-37/year).
# BTC/ETH edge: Elder Ray measures bull/bear strength relative to EMA13; ADX regime avoids whipsaws; volume confirms follow-through.

name = "6h_ElderRay_1dADX_Regime_VolumeSpike"
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
    
    # Calculate EMA13 for Elder Ray (13-period)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Get 1d data for ADX calculation
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
    
    # Align 1d indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate volatility normalization for Elder Ray thresholds (20-period std)
    bull_power_std = pd.Series(bull_power).rolling(window=20, min_periods=20).std().values
    bear_power_std = pd.Series(bear_power).rolling(window=20, min_periods=20).std().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(bull_power_std[i]) or np.isnan(bear_power_std[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.3 x 20-period EMA
        volume_confirm = volume[i] > (1.3 * vol_ema_20[i])
        
        if position == 0:
            # Determine regime: ranging (ADX<20) or trending (ADX>=20)
            if adx_aligned[i] < 20:
                # Ranging market: mean reversion from extreme Elder Ray readings
                # Long when Bull Power is extremely negative (bearish exhaustion)
                # Short when Bear Power is extremely positive (bullish exhaustion)
                if (bull_power[i] < -bull_power_std[i] and 
                    volume_confirm):
                    signals[i] = 0.25
                    position = 1
                elif (bear_power[i] > bear_power_std[i] and 
                      volume_confirm):
                    signals[i] = -0.25
                    position = -1
            else:
                # Trending market: follow Elder Ray momentum
                # Long when Bull Power > 0 (bulls in control)
                # Short when Bear Power < 0 (bears in control)
                if (bull_power[i] > 0 and 
                    volume_confirm):
                    signals[i] = 0.25
                    position = 1
                elif (bear_power[i] < 0 and 
                      volume_confirm):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Bull Power turns negative OR ADX weakening (<15) OR volume drops
            if (bull_power[i] <= 0 or 
                adx_aligned[i] < 15 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power turns positive OR ADX weakening (<15) OR volume drops
            if (bear_power[i] >= 0 or 
                adx_aligned[i] < 15 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals