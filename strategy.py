#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R with volume confirmation and ATR-based trend filter
# Williams %R identifies overbought/oversold conditions (-20 to 0 = overbought, -80 to -100 = oversold)
# In trending markets (ADX > 25): short when %R > -20 with volume confirmation, long when %R < -80 with volume confirmation
# In ranging markets (ADX <= 25): fade extremes - long at %R < -80, short at %R > -20
# Uses discrete position sizing 0.25 to target ~20-50 trades/year and minimize fee drag
# Works in bull/bear markets: follows momentum in trends, mean reverts at extremes in ranges

name = "4h_1d_williamsr_volume_trend_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.zeros_like(close_1d)
    
    # Calculate 1d Williams %R(14)
    def calculate_williams_r(high, low, close, period=14):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        wr = np.where((highest_high - lowest_low) != 0, wr, -50)  # Avoid division by zero
        return wr
    
    wr_1d = calculate_williams_r(high_1d, low_1d, close_1d, 14)
    
    # Calculate 1d ADX(14) for trend regime filter
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
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing
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
        dx = np.where((plus_di + minus_di) != 0, dx, 0)
        
        adx = wilders_smoothing(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate 1d average volume (20-period) for volume confirmation
    vol_s_1d = pd.Series(volume_1d)
    avg_vol_1d = vol_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
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
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = not np.isnan(vol_ma_20[i]) and volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend regime: ADX > 25 = trending, ADX <= 25 = ranging
        trending_regime = adx_1d_aligned[i] > 25
        ranging_regime = adx_1d_aligned[i] <= 25
        
        if position == 1:  # Long position
            if trending_regime and volume_confirmed:
                # Exit long if Williams %R rises above -20 (overbought)
                if wr_1d_aligned[i] > -20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if price moves back above oversold level (mean reversion exit)
                if wr_1d_aligned[i] > -50:  # Exit when returning to neutral
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime and volume_confirmed:
                # Exit short if Williams %R falls below -80 (oversold)
                if wr_1d_aligned[i] < -80:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if price moves back below overbought level (mean reversion exit)
                if wr_1d_aligned[i] < -50:  # Exit when returning to neutral
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime and volume_confirmed:
                # Momentum strategy in trending market
                if wr_1d_aligned[i] < -80:  # Oversold -> long
                    position = 1
                    signals[i] = 0.25
                elif wr_1d_aligned[i] > -20:  # Overbought -> short
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean reversion at extremes in ranging market
                if wr_1d_aligned[i] < -80:  # Deep oversold -> long
                    position = 1
                    signals[i] = 0.25
                elif wr_1d_aligned[i] > -20:  # Deep overbought -> short
                    position = -1
                    signals[i] = -0.25
    
    return signals