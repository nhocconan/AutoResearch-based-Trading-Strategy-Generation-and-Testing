#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d SuperTrend(10,3) as trend filter and 6h ATR-based Donchian(10) breakout with volume confirmation
# SuperTrend on 1d provides robust trend direction that works in both bull and bear markets by adapting to volatility.
# Donchian breakout on 6h captures momentum swings within the 1d trend. Volume confirmation ensures participation.
# Designed for moderate trade frequency (~25-40/year) to balance opportunity and cost on 6h timeframe.
# Uses 6h primary with 1d HTF for trend filter - proven combination for BTC/ETH resilience.

name = "6h_SuperTrend10_3_Donchian10_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for SuperTrend calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d SuperTrend(ATR=10, mult=3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(10)
    atr_1d = pd.Series(tr_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2_1d = (high_1d + low_1d) / 2
    basic_ub = hl2_1d + (3.0 * atr_1d)
    basic_lb = hl2_1d - (3.0 * atr_1d)
    
    # Final Upper and Lower Bands
    final_ub = np.zeros(len(close_1d))
    final_lb = np.zeros(len(close_1d))
    supertrend = np.zeros(len(close_1d))
    trend = np.ones(len(close_1d))  # 1 for uptrend, -1 for downtrend
    
    for i in range(len(close_1d)):
        if i == 0:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            if basic_ub[i] < final_ub[i-1] or close_1d[i-1] > final_ub[i-1]:
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i-1]
                
            if basic_lb[i] > final_lb[i-1] or close_1d[i-1] < final_lb[i-1]:
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i-1]
        
        if i == 0:
            supertrend[i] = final_lb[i]
            trend[i] = 1
        else:
            if supertrend[i-1] == final_ub[i-1]:
                if close_1d[i] <= final_ub[i]:
                    supertrend[i] = final_ub[i]
                else:
                    supertrend[i] = final_lb[i]
                    trend[i] = -1
            else:
                if close_1d[i] >= final_lb[i]:
                    supertrend[i] = final_lb[i]
                else:
                    supertrend[i] = final_ub[i]
                    trend[i] = 1
    
    # Align SuperTrend and trend to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    trend_aligned = align_htf_to_ltf(prices, df_1d, trend)
    
    # Calculate 6h ATR(14) for Donchian and stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_6h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h Donchian channels (10-period)
    donchian_upper = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_lower = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 14  # warmup for ATR(14)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.8x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i]) if i >= 20 else np.mean(volume[:i]) if i > 0 else 0
        volume_spike = volume[i] > (1.8 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_supertrend = supertrend_aligned[i]
        curr_trend = trend_aligned[i]
        curr_atr = atr_6h[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and 1d SuperTrend alignment
            if volume_spike:
                # Bullish entry: price breaks above 6h Donchian upper with 1d uptrend
                if curr_close > curr_upper and curr_trend == 1:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 6h Donchian lower with 1d downtrend
                elif curr_close < curr_lower and curr_trend == -1:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry price OR price breaks 6h Donchian lower
            if curr_close < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_lower:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x ATR above entry OR SuperTrend flips
            elif curr_close >= entry_price + 1.5 * curr_atr:
                signals[i] = 0.10  # reduce position
            elif curr_trend == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry price OR price breaks 6h Donchian upper
            if curr_close > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_upper:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x ATR below entry OR SuperTrend flips
            elif curr_close <= entry_price - 1.5 * curr_atr:
                signals[i] = -0.10  # reduce position
            elif curr_trend == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals