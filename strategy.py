#!/usr/bin/env python3
"""
6h_AsymmetricRegime_ADX_EMA21_v2
Hypothesis: Asymmetric strategy for 6h timeframe using ADX regime detection and EMA21 trend filter. 
In trending markets (ADX > 25): trade pullbacks to EMA21 in trend direction. 
In ranging markets (ADX < 20): fade moves to Bollinger Bands (20,2) with RSI confirmation. 
Uses 12h HTF for regime confirmation to reduce whipsaw. Position size 0.25.
Target: 80-120 total trades over 4 years = 20-30/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 12h for regime detection
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[1:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    period = 14
    tr_period = WilderSmooth(tr, period)
    plus_dm_period = WilderSmooth(plus_dm, period)
    minus_dm_period = WilderSmooth(minus_dm, period)
    
    # Avoid division by zero
    plus_di = np.where(tr_period != 0, (plus_dm_period / tr_period) * 100, 0)
    minus_di = np.where(tr_period != 0, (minus_dm_period / tr_period) * 100, 0)
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = WilderSmooth(dx, period)
    
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate 12h EMA200 for long-term trend filter
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Calculate 6h indicators
    # EMA21 for pullback entries
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Bollinger Bands (20,2) for mean reversion
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma_20 + (bb_std * std_20)
    lower_band = sma_20 - (bb_std * std_20)
    
    # RSI(14) for mean reversion confirmation
    rsi_period = 14
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    if len(gain) >= rsi_period:
        avg_gain[rsi_period-1] = np.nanmean(gain[1:rsi_period])
        avg_loss[rsi_period-1] = np.nanmean(loss[1:rsi_period])
        for i in range(rsi_period, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for the longest indicator
    start_idx = max(50, 200)  # EMA200 needs most bars
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(ema_200_12h_aligned[i]) or
            np.isnan(ema_21[i]) or
            np.isnan(sma_20[i]) or
            np.isnan(std_20[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine regime from 12h ADX
        is_trending = adx_aligned[i] > 25
        is_ranging = adx_aligned[i] < 20
        
        # Long-term trend filter from 12h EMA200
        long_term_uptrend = close[i] > ema_200_12h_aligned[i]
        long_term_downtrend = close[i] < ema_200_12h_aligned[i]
        
        if position == 0:
            if is_trending:
                # Trending regime: pullback to EMA21 in trend direction
                long_setup = (close[i] <= ema_21[i] * 1.005) and long_term_uptrend and (close[i] > ema_21[i])
                short_setup = (close[i] >= ema_21[i] * 0.995) and long_term_downtrend and (close[i] < ema_21[i])
            elif is_ranging:
                # Ranging regime: fade Bollinger Bands with RSI confirmation
                long_setup = (close[i] <= lower_band[i]) and (rsi[i] < 30)
                short_setup = (close[i] >= upper_band[i]) and (rsi[i] > 70)
            else:
                # Transition regime: no trades
                long_setup = False
                short_setup = False
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions
            if is_trending:
                # In trend: exit on trend reversal or extended move
                if not long_term_uptrend or (close[i] >= ema_21[i] * 1.02):
                    signals[i] = 0.0
                    position = 0
            else:
                # In range or transition: exit at mean reversion
                if close[i] >= sma_20[i] or rsi[i] > 50:
                    signals[i] = 0.0
                    position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions
            if is_trending:
                # In trend: exit on trend reversal or extended move
                if not long_term_downtrend or (close[i] <= ema_21[i] * 0.98):
                    signals[i] = 0.0
                    position = 0
            else:
                # In range or transition: exit at mean reversion
                if close[i] <= sma_20[i] or rsi[i] < 50:
                    signals[i] = 0.0
                    position = 0
    
    return signals

name = "6h_AsymmetricRegime_ADX_EMA21_v2"
timeframe = "6h"
leverage = 1.0