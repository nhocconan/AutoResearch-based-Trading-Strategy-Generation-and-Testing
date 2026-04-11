#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_elder_ray_trend_follow_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate EMA13 and EMA26 on daily close
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema26_1d = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # Align EMAs to 6h timeframe
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    ema26_aligned = align_htf_to_ltf(prices, df_1d, ema26_1d)
    
    # Calculate Elder Ray components on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 6-day EMA of bull/bear power for signal smoothing
    bull_power_ema6_1d = pd.Series(bull_power_1d).ewm(span=6, adjust=False, min_periods=6).mean().values
    bear_power_ema6_1d = pd.Series(bear_power_1d).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    bull_power_ema6_aligned = align_htf_to_ltf(prices, df_1d, bull_power_ema6_1d)
    bear_power_ema6_aligned = align_htf_to_ltf(prices, df_1d, bear_power_ema6_1d)
    
    # Calculate ATR for volatility filter (14-period on 1d)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate average volume for volume filter (20-period on 1d)
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 30 to ensure all indicators are valid
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(ema13_aligned[i]) or np.isnan(ema26_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(bull_power_ema6_aligned[i]) or np.isnan(bear_power_ema6_aligned[i]) or
            np.isnan(atr_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Session filter: 00-23 UTC (6h bars, trade all hours but avoid low volatility periods)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        # Trade during major market sessions: Asian (00-08), European (08-16), US (16-24)
        # Avoid only the quietest period: 04-08 UTC
        in_session = not (4 <= hour < 8)
        
        # Volatility filter: current ATR > 0.8 * average ATR (avoid extremely low volatility)
        vol_filter = atr_aligned[i] > 0.8 * np.nanmedian(atr_aligned[max(0, i-20):i])
        
        # Volume filter: current volume > 1.2 * average volume (moderate threshold)
        vol_surge = volume[i] > 1.2 * vol_avg_aligned[i]
        
        # Trend filter: EMA13 > EMA26 for uptrend, EMA13 < EMA26 for downtrend
        uptrend = ema13_aligned[i] > ema26_aligned[i]
        downtrend = ema13_aligned[i] < ema26_aligned[i]
        
        # Elder Ray signals:
        # Bull power > 0 and increasing indicates bullish momentum
        # Bear power < 0 and decreasing indicates bearish momentum
        bullish_momentum = bull_power_aligned[i] > 0 and bull_power_ema6_aligned[i] > bull_power_ema6_aligned[i-1]
        bearish_momentum = bear_power_aligned[i] < 0 and bear_power_ema6_aligned[i] < bear_power_ema6_aligned[i-1]
        
        # Entry conditions
        long_entry = (uptrend and bullish_momentum and vol_filter and vol_surge and in_session)
        short_entry = (downtrend and bearish_momentum and vol_filter and vol_surge and in_session)
        
        # Exit conditions: opposite Elder Ray signal or trend reversal
        exit_long = (not uptrend) or (bear_power_aligned[i] > 0)  # Trend ends or bear power turns positive
        exit_short = (not downtrend) or (bull_power_aligned[i] < 0)  # Trend ends or bull power turns negative
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals