#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with volume confirmation and 1d ADX trend filter
# Donchian channels identify key breakout levels where institutional order flow accelerates.
# Breakouts above upper channel or below lower channel with volume spike indicate strong momentum.
# 1d ADX > 25 ensures alignment with trending market to avoid false breakouts in ranging conditions.
# Designed for low trade frequency (<30/year) to minimize fee drag in both bull and bear markets.
# Uses 12h timeframe as requested, with 1d HTF for Donchian levels and ADX trend filter.

name = "12h_Donchian20_Breakout_1dADXTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Donchian calculation and ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper channel: highest high of last 20 periods
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low of last 20 periods
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (wait for completed 1d bar)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Calculate 1d ADX(14) for trend filter
    # True Range
    tr1 = pd.Series(high_1d[1:]) - pd.Series(low_1d[1:])
    tr2 = np.abs(pd.Series(high_1d[1:]) - pd.Series(close_1d[:-1]))
    tr3 = np.abs(pd.Series(low_1d[1:]) - pd.Series(close_1d[:-1]))
    tr_1d = pd.concat([pd.Series([np.max([tr1.iloc[0] if len(tr1) > 0 else 0, 
                                        tr2.iloc[0] if len(tr2) > 0 else 0, 
                                        tr3.iloc[0] if len(tr3) > 0 else 0])]), 
                       pd.Series(np.maximum(tr1, np.maximum(tr2, tr3)))], ignore_index=True).values
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(high_1d[1:]) - pd.Series(high_1d[:-1])
    dm_minus = pd.Series(low_1d[:-1]) - pd.Series(low_1d[1:])
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0.0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0.0)
    dm_plus = np.concatenate([[0.0], dm_plus])
    dm_minus = np.concatenate([[0.0], dm_minus])
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_smooth = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_smooth
    di_minus = 100 * dm_minus_smooth / atr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 12h timeframe (wait for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate ATR(14) for dynamic stoploss on 12h
    tr1_12h = high[1:] - low[1:]
    tr2_12h = np.abs(high[1:] - close[:-1])
    tr3_12h = np.abs(low[1:] - close[:-1])
    tr_12h = np.concatenate([[np.max([tr1_12h[0], tr2_12h[0], tr3_12h[0]])], 
                             np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))])
    atr_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for ADX(14)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 30-period average
        vol_ma_30 = np.mean(volume[max(0, i-30):i])
        volume_spike = volume[i] > (2.0 * vol_ma_30)
        
        curr_close = close[i]
        curr_upper = upper_aligned[i]
        curr_lower = lower_aligned[i]
        curr_adx = adx_aligned[i]
        curr_atr = atr_12h[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and strong trend (ADX > 25)
            if volume_spike and curr_adx > 25:
                # Bullish entry: price breaks above 1d Donchian upper channel
                if curr_close > curr_upper:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 1d Donchian lower channel
                elif curr_close < curr_lower:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks 1d Donchian lower channel
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_lower:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches midpoint of Donchian channel
            elif curr_close >= (curr_upper + curr_lower) / 2:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks 1d Donchian upper channel
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_upper:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches midpoint of Donchian channel
            elif curr_close <= (curr_upper + curr_lower) / 2:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals