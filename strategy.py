#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA34 trend filter with volume spike confirmation
# Long when Williams %R(14) crosses above -80 (oversold bounce) AND price > 1d EMA34 AND volume > 2.0x 20-bar avg
# Short when Williams %R(14) crosses below -20 (overbought rejection) AND price < 1d EMA34 AND volume > 2.0x 20-bar avg
# Exit when Williams %R returns to -50 (mean reversion) or opposite signal occurs
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 20-50 trades/year on 6h.
# Williams %R identifies exhaustion points in both bull and bear markets; trend filter ensures we trade with higher timeframe momentum.
# Volume confirmation reduces false signals in low-conviction environments.

name = "6h_WilliamsR_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(34) on 1d data
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R(14) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r[highest_high == lowest_low] = -50.0
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Volume MA(20) and Williams %R(14) need lookback
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        wr = williams_r[i]
        ema_34 = ema_34_1d_aligned[i]
        
        # Williams %R crossovers for entry
        wr_prev = williams_r[i-1]
        
        # Long when Williams %R crosses above -80 (oversold bounce) AND price > 1d EMA34 AND volume confirmation
        if wr_prev <= -80.0 and wr > -80.0 and curr_close > ema_34 and vol_conf:
            signals[i] = 0.25
            position = 1
        # Short when Williams %R crosses below -20 (overbought rejection) AND price < 1d EMA34 AND volume confirmation
        elif wr_prev >= -20.0 and wr < -20.0 and curr_close < ema_34 and vol_conf:
            signals[i] = -0.25
            position = -1
        # Exit conditions: Williams %R returns to -50 (mean reversion) or opposite signal
        elif position == 1:  # Long - exit when Williams %R >= -50 or short signal occurs
            if wr >= -50.0 or (wr_prev >= -20.0 and wr < -20.0 and curr_close < ema_34 and vol_conf):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Williams %R <= -50 or long signal occurs
            if wr <= -50.0 or (wr_prev <= -80.0 and wr > -80.0 and curr_close > ema_34 and vol_conf):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals