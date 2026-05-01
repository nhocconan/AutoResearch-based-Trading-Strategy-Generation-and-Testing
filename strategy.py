#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R + 1w EMA trend filter with volume confirmation.
# Uses 1w EMA50 for major trend direction (bull/bear) to capture multi-week regimes.
# Williams %R(14) identifies overbought/oversold conditions for mean reversion entries.
# Long when: 1w EMA50 uptrend AND Williams %R < -80 (oversold) AND volume > 1.5x 20-period average.
# Short when: 1w EMA50 downtrend AND Williams %R > -20 (overbought) AND volume > 1.5x 20-period average.
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 10-20 trades/year.
# Williams %R is a momentum oscillator that measures overbought/oversold levels.
# 1w EMA provides structural trend filter to avoid counter-trend trades in strong markets.
# Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend) by aligning with higher timeframe structure.

name = "1d_WilliamsR_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend direction
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams %R(14) on 1d timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Williams %R and volume MA
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_volume_ratio = volume_ratio[i]
        
        # Trend determination from 1w EMA50
        # Need prior EMA value to determine trend direction
        if i == start_idx:
            prev_ema_50_1w = curr_ema_50_1w
        else:
            prev_ema_50_1w = ema_50_1w_aligned[i-1]
        
        # Determine 1w trend: rising if current EMA > previous EMA
        ema_trend_up = curr_ema_50_1w > prev_ema_50_1w
        ema_trend_down = curr_ema_50_1w < prev_ema_50_1w
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: 1w EMA uptrend AND Williams %R < -80 (oversold) AND volume confirmation
            if (ema_trend_up and 
                curr_williams_r < -80 and 
                curr_volume_ratio > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: 1w EMA downtrend AND Williams %R > -20 (overbought) AND volume confirmation
            elif (ema_trend_down and 
                  curr_williams_r > -20 and 
                  curr_volume_ratio > 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: 1w EMA turns downtrend OR Williams %R > -50 (mean reversion) OR volume dries up
            if (not ema_trend_up or 
                curr_williams_r > -50 or 
                curr_volume_ratio < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: 1w EMA turns uptrend OR Williams %R < -50 (mean reversion) OR volume dries up
            if (not ema_trend_down or 
                curr_williams_r < -50 or 
                curr_volume_ratio < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals