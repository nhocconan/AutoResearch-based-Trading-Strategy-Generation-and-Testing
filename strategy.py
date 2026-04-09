#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extremes with volume confirmation and 1d ADX regime filter
# Williams %R identifies overbought/oversold conditions: > -20 = overbought, < -80 = oversold
# In low ADX (<25) ranging markets: mean reversion at extremes (long at %R < -80, short at %R > -20)
# In high ADX (>25) trending markets: continuation breakouts (long when %R crosses above -80 with volume, short when crosses below -20 with volume)
# Uses discrete position sizing 0.25 to target ~12-37 trades/year and minimize fee drag
# Works in bull/bear markets: mean reversion captures reversals in ranging markets, continuation follows trends in trending markets

name = "6h_1d_williamsr_adx_volume_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R(14)
    def calculate_williams_r(high, low, close, period=14):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        wr = np.where((highest_high - lowest_low) == 0, -50, wr)  # Avoid division by zero
        return wr
    
    wr_1d = calculate_williams_r(high_1d, low_1d, close_1d, 14)
    
    # Calculate 1d ADX(14) for regime filter
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        up_move = np.concatenate([[np.nan], high[1:] - high[:-1]])
        down_move = np.concatenate([[np.nan], low[:-1] - low[1:]])
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed TR, PlusDM, MinusDM using Wilder's smoothing
        def wilders_smoothing(values, period):
            if len(values) < period:
                return np.full(len(values), np.nan)
            alpha = 1.0 / period
            result = np.full(len(values), np.nan)
            result[period-1] = np.nanmean(values[:period])
            for i in range(period, len(values)):
                result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
            return result
        
        tr_smooth = wilders_smoothing(tr, period)
        plus_dm_smooth = wilders_smoothing(plus_dm, period)
        minus_dm_smooth = wilders_smoothing(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # DX and ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        dx = np.where((plus_di + minus_di) == 0, 0, dx)
        
        adx = wilders_smoothing(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate 1d average volume (20-period)
    if 'volume' in df_1d.columns:
        volume_1d = df_1d['volume'].values
    else:
        volume_1d = np.zeros_like(close_1d)
    
    vol_s_1d = pd.Series(volume_1d)
    avg_vol_1d = vol_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe
    wr_1d_aligned = align_htf_to_ltf(prices, df_1d, wr_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(wr_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(avg_vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # Regime filter: ADX < 25 = ranging, ADX > 25 = trending
        ranging_regime = adx_1d_aligned[i] < 25
        trending_regime = adx_1d_aligned[i] > 25
        
        if position == 1:  # Long position
            if ranging_regime:
                # Exit long if Williams %R rises above -50 (mean reversion exit)
                if wr_1d_aligned[i] > -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif trending_regime and volume_confirmed:
                # Exit long if Williams %R falls below -80 (trend exhaustion)
                if wr_1d_aligned[i] < -80:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if ranging_regime:
                # Exit short if Williams %R falls below -50 (mean reversion exit)
                if wr_1d_aligned[i] < -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif trending_regime and volume_confirmed:
                # Exit short if Williams %R rises above -20 (trend exhaustion)
                if wr_1d_aligned[i] > -20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if ranging_regime:
                # Mean reversion at Williams %R extremes
                if wr_1d_aligned[i] < -80:  # Oversold
                    position = 1
                    signals[i] = 0.25
                elif wr_1d_aligned[i] > -20:  # Overbought
                    position = -1
                    signals[i] = -0.25
            elif trending_regime and volume_confirmed:
                # Continuation breakout when Williams %R crosses extreme levels with volume
                if wr_1d_aligned[i] > -80 and wr_1d_aligned[i-1] <= -80:  # Crossed above -80
                    position = 1
                    signals[i] = 0.25
                elif wr_1d_aligned[i] < -20 and wr_1d_aligned[i-1] >= -20:  # Crossed below -20
                    position = -1
                    signals[i] = -0.25
    
    return signals