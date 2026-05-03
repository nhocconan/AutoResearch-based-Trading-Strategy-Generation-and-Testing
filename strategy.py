#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R with 1w EMA50 trend filter and volume confirmation.
# Long when Williams %R crosses above -80 from below in bull trend (close > 1w EMA50) with volume > 1.8x 20-period MA.
# Short when Williams %R crosses below -20 from above in bear trend (close < 1w EMA50) with volume spike.
# Uses discrete position sizing (0.25) to minimize fee churn. 1w EMA50 provides strong trend filter.
# Williams %R identifies overextended conditions for mean reversion within the trend.
# Volume confirmation ensures institutional participation. Target: 30-100 total trades over 4 years (7-25/year).
# Works in both bull and bear markets: trend filter ensures we only trade in direction of 1w momentum,
# while Williams %R provides precise entry points for mean reversion entries.

name = "1d_WilliamsR_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams %R (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume regime: current 1d volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    prev_williams_r = 0  # previous Williams %R value for crossover detection
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(prev_williams_r)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            prev_williams_r = williams_r[i] if not np.isnan(williams_r[i]) else prev_williams_r
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1w_aligned[i]
        wr = williams_r[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Williams %R crossover conditions
        wr_cross_above_80 = prev_williams_r <= -80 and wr > -80
        wr_cross_below_20 = prev_williams_r >= -20 and wr < -20
        
        # Entry logic
        if position == 0:
            if is_bull_trend and wr_cross_above_80 and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and wr_cross_below_20 and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 OR trend reversal
            if wr < -50 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 OR trend reversal
            if wr > -50 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        
        prev_williams_r = wr
    
    return signals