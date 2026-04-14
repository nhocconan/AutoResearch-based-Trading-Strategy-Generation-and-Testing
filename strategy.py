#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy with 4h ADX trend filter and 1d Bollinger Bands reversal
# ADX > 25 indicates trending market (use for trend direction)
# Bollinger Bands %B < 0.2 or > 0.8 indicates oversold/overbought conditions for mean reversion entries
# Works in both bull and bear markets: trend filter prevents counter-trend trades in strong moves,
# while Bollinger Bands capture mean reversion within the trend
# Uses 4h ADX for trend regime and 1d BB% for entry timing - avoids overtrading

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 4h data ONCE for ADX
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h ADX (14 periods)
    adx_len = 14
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=adx_len, min_periods=adx_len).sum().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_sum / tr_sum
    minus_di = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=adx_len, min_periods=adx_len).mean().values
    
    # Align ADX to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Load 1d data ONCE for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Bollinger Bands (20, 2)
    bb_len = 20
    bb_mult = 2.0
    bb_src = df_1d['close'].values
    
    # Basis (SMA)
    basis = pd.Series(bb_src).rolling(window=bb_len, min_periods=bb_len).mean().values
    # Deviation
    dev = bb_mult * pd.Series(bb_src).rolling(window=bb_len, min_periods=bb_len).std().values
    # Upper and Lower bands
    upper = basis + dev
    lower = basis - dev
    # Percent B (%B)
    bb_pctb = (bb_src - lower) / (upper - lower)
    bb_pctb = np.where((upper - lower) == 0, 0.5, bb_pctb)  # Avoid division by zero
    
    # Align BB %B to 1h timeframe
    bb_pctb_aligned = align_htf_to_ltf(prices, df_1d, bb_pctb)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = max(100, adx_len + bb_len)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(bb_pctb_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Mean reversion signals from Bollinger Bands %B
        oversold = bb_pctb_aligned[i] < 0.2
        overbought = bb_pctb_aligned[i] > 0.8
        
        if position == 0:
            # Enter long: trending + oversold (pullback in uptrend)
            if trending and oversold:
                # Additional filter: price above 20-period SMA for uptrend confirmation
                sma_20 = pd.Series(close[:i+1]).rolling(window=20, min_periods=20).mean().iloc[-1]
                if price > sma_20:
                    position = 1
                    signals[i] = position_size
            # Enter short: trending + overbought (pullback in downtrend)
            elif trending and overbought:
                # Additional filter: price below 20-period SMA for downtrend confirmation
                sma_20 = pd.Series(close[:i+1]).rolling(window=20, min_periods=20).mean().iloc[-1]
                if price < sma_20:
                    position = -1
                    signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above 50% BB level (mean reversion complete) OR ADX drops
            if bb_pctb_aligned[i] > 0.5 or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below 50% BB level (mean reversion complete) OR ADX drops
            if bb_pctb_aligned[i] < 0.5 or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4hADX_1dBB_PB_MeanReversion_v1"
timeframe = "1h"
leverage = 1.0