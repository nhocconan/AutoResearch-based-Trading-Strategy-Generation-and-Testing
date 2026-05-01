#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1w EMA50 trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) AND 1w close > EMA50 (bullish trend) AND volume > 1.5x 20-bar average.
# Short when Williams %R > -20 (overbought) AND 1w close < EMA50 (bearish trend) AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Williams %R identifies exhaustion points; 1w EMA50 provides major trend filter; volume confirms conviction.
# Primary timeframe: 6h, HTF: 1w for trend filter.

name = "6h_WilliamsR_Reversal_1wEMA50_Trend_VolumeConfirm_v1"
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
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    # 1w EMA50 calculation
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate 1w close aligned for trend bias
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Williams %R calculation (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current 6h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Williams %R and EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_aligned[i]) or np.isnan(close_1w_aligned[i]) or \
           np.isnan(williams_r[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)  # Volume spike threshold
        
        # Williams %R signals
        oversold = williams_r[i] < -80  # Oversold condition
        overbought = williams_r[i] > -20  # Overbought condition
        
        # Trend filter: use 1w close vs its EMA50 for bias
        bullish_bias = close_1w_aligned[i] > ema_aligned[i]  # 1w close above its EMA50 = bullish
        bearish_bias = close_1w_aligned[i] < ema_aligned[i]  # 1w close below its EMA50 = bearish
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: oversold AND bullish bias AND volume confirmation
            if (oversold and 
                bullish_bias and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: overbought AND bearish bias AND volume confirmation
            elif (overbought and 
                  bearish_bias and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses above -50 (momentum shift) OR bearish bias (trend change)
            if (williams_r[i] > -50 or 
                bearish_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -50 (momentum shift) OR bullish bias (trend change)
            if (williams_r[i] < -50 or 
                bullish_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals