#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout + 1w ADX regime + volume confirmation
# In trending markets (ADX>=25), we trade breakouts in trend direction: long on R3 breakout in uptrend, short on S3 breakout in downtrend.
# In ranging markets (ADX<25), we fade extremes: short near R3, long near S3.
# Volume confirmation (>1.5x 20-period EMA) reduces false breakouts. Designed for 12h timeframe targeting 50-150 total trades over 4 years.
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "12h_Camarilla_R3S3_1wADX_Regime_Volume"
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
    
    # Get 1w data for ADX and Camarilla pivot levels
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
    
    # Calculate 1w Camarilla pivot levels
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4)
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    camarilla_r3 = close_1w + ((high_1w - low_1w) * 1.1 / 4)
    camarilla_s3 = close_1w - ((high_1w - low_1w) * 1.1 / 4)
    
    # Align 1w indicators to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx.values)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
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
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
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
                # Trend direction: +DI > -DI indicates uptrend
                plus_di_1w = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / atr)
                minus_di_1w = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / atr)
                plus_di_aligned = align_htf_to_ltf(prices, df_1w, plus_di_1w.values)
                minus_di_aligned = align_htf_to_ltf(prices, df_1w, minus_di_1w.values)
                
                # Long: R3 breakout in uptrend (+DI > -DI)
                if (close[i] > camarilla_r3_aligned[i] and 
                    volume_confirm and 
                    plus_di_aligned[i] > minus_di_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: S3 breakout in downtrend (-DI > +DI)
                elif (close[i] < camarilla_s3_aligned[i] and 
                      volume_confirm and 
                      minus_di_aligned[i] > plus_di_aligned[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price retouches midpoint between R3 and S3 OR ADX weakening (<20) OR volume drops
            mid = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2
            if (close[i] <= mid or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches midpoint between R3 and S3 OR ADX weakening (<20) OR volume drops
            mid = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2
            if (close[i] >= mid or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals