#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d ATR-based volatility regime filter + Donchian(20) breakout with volume confirmation
# In bear markets, volatility often precedes directional moves. ATR(30) > ATR(60) indicates expanding volatility regime.
# Donchian(20) breakout captures the move, volume confirmation ensures institutional participation.
# 4h EMA(50) trend filter avoids counter-trend trades. Designed for low trade frequency to minimize fee drag.

name = "4h_ATR_Vol_Regime_Donchian20_Breakout_4hEMA50_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d ATR(30) and ATR(60) for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_30 = pd.Series(tr_1d).ewm(span=30, adjust=False, min_periods=30).mean().values
    atr_60 = pd.Series(tr_1d).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Volatility regime: ATR(30) > ATR(60) indicates expanding volatility
    vol_regime = atr_30 > atr_60
    
    # Align volatility regime to 4h timeframe
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # Calculate 4h Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate 4h EMA(50) for trend filter
    close_s = pd.Series(close)
    ema_50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(60, 50, 20, 14)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.8x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (1.8 * vol_ma_20)
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_50[i]
        curr_atr = atr[i]
        curr_vol_regime = vol_regime_aligned[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volatility regime and volume spike
            if curr_vol_regime and volume_spike:
                # Bullish entry: price breaks above Donchian upper band with 4h uptrend
                if curr_close > curr_highest_high and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below Donchian lower band with 4h downtrend
                elif curr_close < curr_lowest_low and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks Donchian lower band
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_lowest_low:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches middle of Donchian channel
            elif curr_close >= (curr_highest_high + curr_lowest_low) / 2:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks Donchian upper band
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_highest_high:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches middle of Donchian channel
            elif curr_close <= (curr_highest_high + curr_lowest_low) / 2:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals