#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA34 trend filter and volume confirmation
# Williams Alligator (jaw/teeth/lips) identifies trend absence/presence via smoothed MAs
# 1w EMA34 ensures alignment with weekly trend; volume >1.5x confirms participation
# Discrete sizing (0.25) minimizes fee churn; target 30-100 total trades over 4 years

name = "1d_WilliamsAlligator_1wEMA34_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR for volatility (14-period)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume > (1.5 * vol_ma_50)
    
    # Williams Alligator: jaw (13), teeth (8), lips (5) - all smoothed with future shift
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean().shift(5).values
    lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean().shift(3).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(jaw_period + 8, teeth_period + 5, lips_period + 3, 34, 50)
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(atr[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma_50[i]) or
            np.isnan(jaw[i]) or
            np.isnan(teeth[i]) or
            np.isnan(lips[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish entry: lips > teeth > jaw (aligned) + price > weekly EMA34
                if lips[i] > teeth[i] > jaw[i] and curr_close > curr_ema_34_1w:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: lips < teeth < jaw (aligned) + price < weekly EMA34
                elif lips[i] < teeth[i] < jaw[i] and curr_close < curr_ema_34_1w:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: lips < teeth (Alligator sleeping) or price crosses below weekly EMA34
            if lips[i] < teeth[i] or curr_close < curr_ema_34_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: lips > teeth (Alligator sleeping) or price crosses above weekly EMA34
            if lips[i] > teeth[i] or curr_close > curr_ema_34_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals