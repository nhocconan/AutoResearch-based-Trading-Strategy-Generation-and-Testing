#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d EMA34 trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In ranging markets (2025+),
# mean reversion from extremes works well. Trend filter ensures we only take reversals
# in direction of higher timeframe trend to avoid catching falling knives.
# Long: %R < -80 (oversold) AND price > 1d EMA34 (uptrend) AND volume > 1.5x 20-bar avg.
# Short: %R > -20 (overbought) AND price < 1d EMA34 (downtrend) AND volume > 1.5x 20-bar avg.
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 80-120 total trades over 4 years (20-30/year).

name = "6h_WilliamsR_Reversal_1dEMA34_Trend_VolumeConfirm_v1"
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
    open_time = prices['open_time']
    
    # Pre-compute session hours for efficiency (00-24 UTC - trade all hours for 6h)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 calculation
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # 1d trend: price above/below EMA34
    price_above_ema = close > ema_34_aligned
    price_below_ema = close < ema_34_aligned
    
    # Williams %R calculation (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    williams_r = np.where(hl_range != 0, (highest_high - close) / hl_range * -100, -50)
    
    # Volume confirmation: current 6h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 14) + 5  # warmup for EMA, volume MA, and Williams %R
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if vol_ma[i] <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = volume[i] > (vol_ma[i] * 1.5)  # Volume spike threshold
        
        # Williams %R signals
        oversold = williams_r[i] < -80  # Oversold condition
        overbought = williams_r[i] > -20  # Overbought condition
        
        # Exit conditions: %R returns to neutral zone (-50 center)
        exit_long = williams_r[i] > -50  # Return from oversold
        exit_short = williams_r[i] < -50  # Return from overbought
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: oversold AND uptrend AND volume confirmation
            if (oversold and 
                price_above_ema[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: overbought AND downtrend AND volume confirmation
            elif (overbought and 
                  price_below_ema[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: %R returns to neutral OR trend fails
            if (exit_long or 
                not price_above_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: %R returns to neutral OR trend fails
            if (exit_short or 
                not price_below_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals