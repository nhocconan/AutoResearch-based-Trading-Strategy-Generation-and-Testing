#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Mean Reversion + 1d ADX Regime + Volume Spike
# Williams %R identifies overbought/oversold conditions. In ranging markets (ADX<25), 
# we fade extremes: long when %R < -80, short when %R > -20. In trending markets (ADX>=25),
# we only take trades in trend direction: long when %R < -50 and ADX up, short when %R > -50 and ADX down.
# Volume spike (>2x 20-period EMA) confirms momentum. Designed for 6h timeframe targeting 50-150 total trades.
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown in both bull and bear markets.

name = "6h_WilliamsR_MeanReversion_1dADX_Regime_Volume"
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
    
    # Get 1d data for Williams %R and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    hh_1d = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    williams_r = -100 * (hh_1d - close_1d) / (hh_1d - ll_1d)
    
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
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirm = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Determine regime: ranging (ADX<25) or trending (ADX>=25)
            if adx_aligned[i] < 25:
                # Ranging market: mean reversion at extremes
                if williams_r_aligned[i] < -80 and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                elif williams_r_aligned[i] > -20 and volume_confirm:
                    signals[i] = -0.25
                    position = -1
            else:
                # Trending market: only trade with trend
                # Trend direction: +DI > -DI indicates uptrend
                plus_di_1d = 100 * (pd.Series(df_1d['high']).diff().where(
                    lambda x: (x > pd.Series(df_1d['low']).diff().mul(-1)) & (x > 0), 0.0
                ).rolling(window=14, min_periods=14).sum() / 
                pd.Series(
                    pd.concat([
                        pd.Series(df_1d['high']).sub(df_1d['low']),
                        pd.Series(df_1d['high']).sub(df_1d['close'].shift(1)).abs(),
                        pd.Series(df_1d['low']).sub(df_1d['close'].shift(1)).abs()
                    ], axis=1).max(axis=1)
                ).rolling(window=14, min_periods=14).mean())
                plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di_1d.values)
                minus_di_1d = 100 * (pd.Series(df_1d['low']).diff().mul(-1).where(
                    lambda x: (x > pd.Series(df_1d['high']).diff()) & (x > 0), 0.0
                ).rolling(window=14, min_periods=14).sum() / 
                pd.Series(
                    pd.concat([
                        pd.Series(df_1d['high']).sub(df_1d['low']),
                        pd.Series(df_1d['high']).sub(df_1d['close'].shift(1)).abs(),
                        pd.Series(df_1d['low']).sub(df_1d['close'].shift(1)).abs()
                    ], axis=1).max(axis=1)
                ).rolling(window=14, min_periods=14).mean())
                minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di_1d.values)
                
                # Long: Williams %R < -50 (pullback in uptrend) + volume + uptrend (+DI > -DI)
                if (williams_r_aligned[i] < -50 and 
                    volume_confirm and 
                    plus_di_aligned[i] > minus_di_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R > -50 (pullback in downtrend) + volume + downtrend (-DI > +DI)
                elif (williams_r_aligned[i] > -50 and 
                      volume_confirm and 
                      minus_di_aligned[i] > plus_di_aligned[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Williams %R > -20 (overbought) OR ADX weakening (<20) OR volume drops
            if (williams_r_aligned[i] > -20 or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R < -80 (oversold) OR ADX weakening (<20) OR volume drops
            if (williams_r_aligned[i] < -80 or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals