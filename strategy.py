#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout + 1d volume spike + ADX regime filter
# In trending markets (1d ADX>=25), trade breakouts in trend direction: long on R3 breakout in uptrend, short on S3 breakdown in downtrend.
# In ranging markets (1d ADX<25), fade extremes: short near R3, long near S3.
# Volume confirmation (>2.0x 20-period EMA) reduces false signals. Designed for 4h timeframe targeting 75-200 total trades over 4 years.
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "4h_Camarilla_R3S3_1dADX_Regime_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX and Camarilla levels
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
    
    # Calculate 1d Camarilla levels (R3, S3, midpoint)
    prev_close = df_1d['close'].shift(1)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    camarilla_mid = (camarilla_r3 + camarilla_s3) / 2
    
    # Align 1d indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid.values)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirm = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Determine regime: ranging (ADX<25) or trending (ADX>=25)
            if adx_aligned[i] < 25:
                # Ranging market: fade extremes (mean reversion)
                if close[i] <= camarilla_s3_aligned[i] and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= camarilla_r3_aligned[i] and volume_confirm:
                    signals[i] = -0.25
                    position = -1
            else:
                # Trending market: trade breakouts in trend direction
                # Calculate 1d +DI and -DI for trend direction
                plus_di_1d = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / atr)
                minus_di_1d = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / atr)
                plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di_1d.values)
                minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di_1d.values)
                
                # Long: R3 breakout in uptrend (+DI > -DI)
                if (close[i] > camarilla_r3_aligned[i] and 
                    volume_confirm and 
                    plus_di_aligned[i] > minus_di_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: S3 breakdown in downtrend (-DI > +DI)
                elif (close[i] < camarilla_s3_aligned[i] and 
                      volume_confirm and 
                      minus_di_aligned[i] > plus_di_aligned[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price retouches midpoint OR ADX weakening (<20) OR volume drops below EMA
            if (close[i] <= camarilla_mid_aligned[i] or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches midpoint OR ADX weakening (<20) OR volume drops below EMA
            if (close[i] >= camarilla_mid_aligned[i] or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals