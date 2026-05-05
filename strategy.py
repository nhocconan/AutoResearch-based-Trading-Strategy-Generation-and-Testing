#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme with 1d trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) AND 1d close > 1d EMA34 AND volume > 1.5x 20-period average
# Short when Williams %R > -20 (overbought) AND 1d close < 1d EMA34 AND volume > 1.5x 20-period average
# Exit when Williams %R crosses above -50 (for long) or below -50 (for short)
# Uses 6h primary timeframe with 1d HTF for trend filter and Williams %R calculation
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Williams %R identifies exhaustion points; 1d EMA34 filters for higher-timeframe trend; volume confirms institutional participation

name = "6h_WilliamsR_EXTREME_1dEMA34_Trend_Volume"
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
    
    # Get 1d data ONCE before loop for trend filter and Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R on 1d data: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Williams %R = -100 * (HHV - Close) / (HHV - LLV)
    # Where HHV = highest high over period, LLV = lowest low over period
    lookback_period = 14
    hhv = pd.Series(df_1d['high'].values).rolling(window=lookback_period, min_periods=lookback_period).max().values
    llv = pd.Series(df_1d['low'].values).rolling(window=lookback_period, min_periods=lookback_period).min().values
    # Avoid division by zero
    denominator = hhv - llv
    williams_r_1d = np.where(denominator != 0, -100 * (hhv - df_1d['close'].values) / denominator, -50)
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(williams_r_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND 1d close > 1d EMA34 AND volume spike
            if (williams_r_1d_aligned[i] < -80 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND 1d close < 1d EMA34 AND volume spike
            elif (williams_r_1d_aligned[i] > -20 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (momentum weakening)
            if williams_r_1d_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (momentum weakening)
            if williams_r_1d_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals