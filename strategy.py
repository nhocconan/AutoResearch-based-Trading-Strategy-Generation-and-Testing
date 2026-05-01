#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d trend filter and volume confirmation.
# Long when Williams %R crosses above -80 from below AND price > 1d EMA50 (uptrend) AND volume > 1.5x 20-bar average.
# Short when Williams %R crosses below -20 from above AND price < 1d EMA50 (downtrend) AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Williams %R identifies overbought/oversold conditions for mean reversion in ranging markets.
# 1d EMA50 filter ensures alignment with higher timeframe trend, reducing counter-trend trades.
# Volume confirmation adds conviction to reversal signals.
# Primary timeframe: 6h, HTF: 1d for trend filter.

name = "6h_WilliamsR_Reversal_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Williams %R on 6h data (period=14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current 6h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Williams %R and EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(williams_r[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        curr_williams_r = williams_r[i]
        curr_ema50 = ema50_1d_aligned[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)  # Volume confirmation threshold
        
        # Williams %R reversal signals
        williams_r_oversold = curr_williams_r < -80  # Oversold condition
        williams_r_overbought = curr_williams_r > -20  # Overbought condition
        
        # Trend filter from 1d EMA50
        uptrend = curr_close > curr_ema50
        downtrend = curr_close < curr_ema50
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold AND uptrend AND volume confirmation
            if (williams_r_oversold and 
                uptrend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought AND downtrend AND volume confirmation
            elif (williams_r_overbought and 
                  downtrend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R becomes overbought OR trend turns down
            if (curr_williams_r > -20 or 
                not uptrend):  # price < EMA50
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R becomes oversold OR trend turns up
            if (curr_williams_r < -80 or 
                not downtrend):  # price > EMA50
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals