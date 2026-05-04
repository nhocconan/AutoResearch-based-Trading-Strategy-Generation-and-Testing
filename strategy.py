#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 12h trend filter and volume confirmation
# Long when Williams %R crosses above -80 from oversold AND 12h bullish trend (close > EMA50) AND volume > 1.5x 20-period volume EMA
# Short when Williams %R crosses below -20 from overbought AND 12h bearish trend (close < EMA50) AND volume > 1.5x 20-period volume EMA
# Williams %R identifies exhaustion points; 12h trend filter ensures alignment with higher timeframe momentum
# Volume confirmation reduces false signals. Targets 12-30 trades/year on 6h timeframe.
# Works in bull markets via buying dips in uptrend and bear markets via selling rallies in downtrend.

name = "6h_WilliamsR_12hTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish_12h = close_12h > ema_50_12h
    trend_bearish_12h = close_12h < ema_50_12h
    
    # Align 12h trend to 6h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_12h, trend_bullish_12h.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_12h, trend_bearish_12h.astype(float))
    
    # Calculate Williams %R (14-period) on 6h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Calculate Williams %R crossover signals
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = np.nan
    
    # Long signal: Williams %R crosses above -80 (oversold recovery)
    williams_long_cross = (williams_r > -80) & (williams_r_prev <= -80)
    # Short signal: Williams %R crosses below -20 (overbought rejection)
    williams_short_cross = (williams_r < -20) & (williams_r_prev >= -20)
    
    # Align Williams %R crossover signals to ensure they're based on completed candles
    williams_long_aligned = align_htf_to_ltf(prices, prices, williams_long_cross.astype(float))
    williams_short_aligned = align_htf_to_ltf(prices, prices, williams_short_cross.astype(float))
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(williams_long_aligned[i]) or np.isnan(williams_short_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 AND 12h bullish trend AND volume spike
            if (williams_long_aligned[i] > 0.5 and 
                trend_bullish_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -20 AND 12h bearish trend AND volume spike
            elif (williams_short_aligned[i] > 0.5 and 
                  trend_bearish_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses below -50 (momentum loss) OR 12h trend turns bearish
            if (williams_r[i] < -50) or (trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses above -50 (momentum loss) OR 12h trend turns bullish
            if (williams_r[i] > -50) or (trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals