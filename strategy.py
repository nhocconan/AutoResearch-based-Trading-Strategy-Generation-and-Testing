#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using Williams %R extreme readings combined with 12h ADX trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; extreme readings (>80 or <20) with reversal candlesticks
# provide high-probability mean reversion entries. 12h ADX > 25 ensures we trade only in trending markets
# to avoid whipsaws in ranging conditions. Volume confirmation filters low-conviction moves.
# Designed for low trade frequency (<30/year) to minimize fee drag while capturing meaningful reversals
# in both bull and bear markets. Uses 6h timeframe as requested, with 12h HTF for ADX trend filter.

name = "6h_WilliamsR_Extreme_12hADX_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ADX(14) for trend filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where(np.isnan(dx), 0, dx)
    adx_12h = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe (wait for completed 12h bar)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate Williams %R(14) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 14  # warmup for Williams %R
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.8x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (1.8 * vol_ma_20)
        
        curr_close = close[i]
        curr_wr = williams_r[i]
        curr_adx = adx_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and strong trend (ADX > 25)
            if volume_spike and curr_adx > 25:
                # Bullish entry: Williams %R crosses above -80 from oversold
                if curr_wr > -80 and williams_r[i-1] <= -80:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Williams %R crosses below -20 from overbought
                elif curr_wr < -20 and williams_r[i-1] >= -20:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR Williams %R overbought (> -20)
            tr = np.maximum(high[i] - low[i], np.maximum(np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1])))
            if i == 0:
                atr_val = tr
            else:
                atr_val = 0.9 * atr_val + 0.1 * tr if 'atr_val' in locals() else tr
            if curr_close < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            elif curr_wr > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR Williams %R oversold (< -80)
            tr = np.maximum(high[i] - low[i], np.maximum(np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1])))
            if i == 0:
                atr_val = tr
            else:
                atr_val = 0.9 * atr_val + 0.1 * tr if 'atr_val' in locals() else tr
            if curr_close > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            elif curr_wr < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals