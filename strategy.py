#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA34 trend filter and ATR-based volume regime.
# Long when Williams %R crosses above -80 (oversold bounce) in bull trend (close > EMA34) with elevated volume.
# Short when Williams %R crosses below -20 (overbought rejection) in bear trend (close < EMA34) with elevated volume.
# Uses discrete position sizing (0.25) and trend-following exits to capture swings in both bull and bear markets.
# Williams %R is effective in ranging markets; EMA34 filter ensures alignment with higher timeframe momentum.
# Volume spike confirms participation. Designed for 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsR_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R on 6h data (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Avoid division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume regime: current 6h volume > 1.8x 20-period MA (slightly looser to increase trade frequency reasonably)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
        wr = williams_r[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Williams %R cross conditions (using prior bar to avoid look-ahead)
        wr_prev = williams_r[i-1] if i > 0 else -50
        wr_cross_above_80 = wr_prev <= -80 and wr > -80   # Oversold bounce
        wr_cross_below_20 = wr_prev >= -20 and wr < -20   # Overbought rejection
        
        # Entry logic
        if position == 0:
            if is_bull_trend and wr_cross_above_80 and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and wr_cross_below_20 and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend reversal or overbought condition
            if close_val < ema_trend or wr >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend reversal or oversold condition
            if close_val > ema_trend or wr <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals