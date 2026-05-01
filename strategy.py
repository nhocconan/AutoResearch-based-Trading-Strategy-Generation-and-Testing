#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h EMA50 trend filter and volume spike confirmation.
# Long when Williams %R < -80 (oversold) AND 12h close > EMA50 (bullish trend) AND volume > 1.8x 20-bar average.
# Short when Williams %R > -20 (overbought) AND 12h close < EMA50 (bearish trend) AND volume > 1.8x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Williams %R captures short-term extremes, EMA50 filters trend alignment, volume spike confirms momentum.
# Primary timeframe: 6h, HTF: 12h for EMA trend filter.

name = "6h_WilliamsR_12hEMA50_Trend_VolumeSpike_v1"
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
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Williams %R (14-period) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # 12h EMA50 trend filter
    prev_close = df_12h['close'].values
    ema_50 = pd.Series(prev_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume confirmation: current 6h volume > 1.8x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA and indicators
    
    for i in range(start_idx, n):
        if np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.8)  # Volume spike threshold
        
        # Williams %R signals
        oversold = curr_williams_r < -80  # oversold condition
        overbought = curr_williams_r > -20  # overbought condition
        
        # Trend filter: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = curr_close > ema_50_aligned[i]
        bearish_trend = curr_close < ema_50_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: oversold AND bullish trend AND volume confirmation
            if (oversold and 
                bullish_trend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: overbought AND bearish trend AND volume confirmation
            elif (overbought and 
                  bearish_trend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses above -50 (mean reversion) OR trend turns bearish
            if (curr_williams_r > -50 or 
                bearish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -50 (mean reversion) OR trend turns bullish
            if (curr_williams_r < -50 or 
                bullish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals