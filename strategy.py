#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA50 trend filter and volume confirmation.
# Long when %R crosses above -20 (from oversold) in bull trend (close > 1d EMA50) with volume > 1.5x 24-period MA.
# Short when %R crosses below -80 (from overbought) in bear trend (close < 1d EMA50) with volume spike.
# Uses 12h primary timeframe with 1d HTF for trend filter and Williams %R levels.
# Williams %R identifies reversals; 1d EMA50 filters counter-trend whipsaw.
# Target: 75-200 total trades over 4 years (19-50/year) with Sharpe > 0 on BTC/ETH/SOL.

name = "12h_WilliamsR_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_1d['close'].values) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 12h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume regime: current 12h volume > 1.5x 24-period MA
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1d_aligned[i]
        williams_r_val = williams_r_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Williams %R levels
        oversold = -80
        overbought = -20
        
        # Entry logic
        if position == 0:
            if is_bull_trend and williams_r_val > oversold and williams_r_aligned[i-1] <= oversold and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and williams_r_val < overbought and williams_r_aligned[i-1] >= overbought and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: %R crosses below -50 (momentum loss) OR trend reversal
            if williams_r_val < -50 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: %R crosses above -50 (momentum loss) OR trend reversal
            if williams_r_val > -50 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals