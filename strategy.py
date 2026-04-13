#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h primary with 1d HTF - Williams %R mean reversion + 1d trend filter + volume confirmation
    # Works in bull/bear by fading extremes in ranging markets (Williams %R) while respecting 1d trend
    # Volume confirmation reduces false signals. Target: 50-150 total trades over 4 years (12-37/year)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF Williams %R, EMA trend, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 1d Williams %R (14-period): %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    def calculate_williams_r(high, low, close, window=14):
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max()
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min()
        williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
        # Replace division by zero with -50 (neutral)
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
        return williams_r.values
    
    williams_r_1d = calculate_williams_r(high_1d, low_1d, close_1d, window=14)
    
    # Calculate 1d EMA (50-period) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume average (20-period)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h primary timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.3x 1d average volume (scaled)
        # Scale 1d average to 6h: approximate by dividing by 4 (4x 6h in 1d)
        volume_confirmed = volume[i] > 1.3 * (vol_avg_20_aligned[i] / 4.0)
        
        # Mean reversion conditions from Williams %R
        oversold = williams_r_aligned[i] < -80  # Extremely oversold
        overbought = williams_r_aligned[i] > -20  # Extremely overbought
        
        # Trend filter: only take longs in uptrend, shorts in downtrend
        uptrend = close_1d[i] > ema_50_aligned[i]  # Use 1d close vs 1d EMA for trend
        downtrend = close_1d[i] < ema_50_aligned[i]
        
        # Entry conditions
        enter_long = oversold and uptrend and volume_confirmed
        enter_short = overbought and downtrend and volume_confirmed
        
        # Exit conditions: Williams %R returns to neutral zone (-50) or opposite extreme
        exit_long = position == 1 and (williams_r_aligned[i] > -50 or williams_r_aligned[i] < -90)
        exit_short = position == -1 and (williams_r_aligned[i] < -50 or williams_r_aligned[i] > -10)
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_williamsr_meanrev_trend_volume_v1"
timeframe = "6h"
leverage = 1.0