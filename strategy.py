#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d trend filter and volume confirmation
# - Williams %R(14) identifies overbought/oversold conditions
# - 1d EMA(50) defines trend direction (long when price > EMA50, short when price < EMA50)
# - 1d volume > 1.3x 20-period average for conviction
# - Entry: Williams %R < -80 in uptrend for long, > -20 in downtrend for short
# - Exit: Williams %R > -20 for long, < -80 for short or trend reversal
# - Session filter: only trade 00:00-12:00 UTC to align with 12h candle close
# - Position size: 0.25 (25%) to balance return and drawdown
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 15-30 trades/year to avoid excessive fee drag

name = "12h_WilliamsR_1dTrend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Williams %R(14) on 12h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(-50).values
    
    # Pre-compute session filter (00:00-12:00 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 12)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1d volume > 1.3x average
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.3 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Look for long entry: uptrend (price > 1d EMA50) + oversold Williams %R + volume
            if close[i] > ema_50_1d_aligned[i] and williams_r[i] < -80 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend (price < 1d EMA50) + overbought Williams %R + volume
            elif close[i] < ema_50_1d_aligned[i] and williams_r[i] > -20 and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on overbought Williams %R or trend reversal
            if williams_r[i] > -20 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on oversold Williams %R or trend reversal
            if williams_r[i] < -80 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals