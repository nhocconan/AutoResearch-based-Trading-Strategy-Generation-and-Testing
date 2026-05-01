#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d EMA50 trend filter and volume spike confirmation.
# Long when Williams %R crosses above -80 (oversold reversal) AND 1d close > EMA50 (bullish trend) AND volume > 1.8x 20-bar average.
# Short when Williams %R crosses below -20 (overbought reversal) AND 1d close < EMA50 (bearish trend) AND volume > 1.8x 20-bar average.
# Williams %R captures mean reversion in 6h swings, EMA50 filters trend alignment, volume spike confirms momentum conviction.
# Designed for low trade frequency (12-37/year) to minimize fee drag while maintaining edge in both bull and bear markets via trend-filtered reversals.

name = "6h_WilliamsR_EMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams %R (14-period) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Avoid division by zero
    hh_ll = highest_high - lowest_low
    williams_r = np.where(hh_ll != 0, (highest_high - close) / hh_ll * -100, -50.0)
    
    # 1d EMA50 trend filter
    prev_close = df_1d['close'].values
    ema_50 = pd.Series(prev_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: current 6h volume > 1.8x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Williams %R and EMA
    
    for i in range(start_idx, n):
        if np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.8)  # Volume spike threshold
        
        # Williams %R reversal signals
        wr_cross_up = williams_r[i] > -80 and williams_r[i-1] <= -80  # cross above -80 (oversold)
        wr_cross_down = williams_r[i] < -20 and williams_r[i-1] >= -20  # cross below -20 (overbought)
        
        # Trend filter: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = curr_close > ema_50_aligned[i]
        bearish_trend = curr_close < ema_50_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 AND bullish trend AND volume confirmation
            if (wr_cross_up and 
                bullish_trend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND bearish trend AND volume confirmation
            elif (wr_cross_down and 
                  bearish_trend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (momentum loss) OR trend turns bearish
            if (williams_r[i] < -50 and williams_r[i-1] >= -50) or bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (momentum loss) OR trend turns bullish
            if (williams_r[i] > -50 and williams_r[i-1] <= -50) or bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals