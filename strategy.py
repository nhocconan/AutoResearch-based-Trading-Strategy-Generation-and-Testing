#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume spike confirmation.
    # Long when: 1d close > 1d EMA50 (bullish trend) AND 6h Williams %R < -80 (oversold) AND 6h volume > 2.0x 20-period MA.
    # Short when: 1d close < 1d EMA50 (bearish trend) AND 6h Williams %R > -20 (overbought) AND 6h volume > 2.0x 20-period MA.
    # Exit when Williams %R reverts to midpoint (-50) or opposite extreme.
    # Uses Williams %R for mean reversion timing, 1d EMA for trend filter, volume spike for confirmation.
    # Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 6h data for Williams %R calculation and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Williams %R calculation (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_6h) / (highest_high_14 - lowest_low_14) * -100
    
    # Volume MA for confirmation
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align all 6h indicators to LTF
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    close_6h_aligned = align_htf_to_ltf(prices, df_6h, close_6h)
    volume_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(volume_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d close vs EMA50
        bullish_trend = close_6h_aligned[i] > ema_50_1d_aligned[i]  # Using 6h close vs 1d EMA50
        bearish_trend = close_6h_aligned[i] < ema_50_1d_aligned[i]
        
        # Williams %R conditions
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        revert_to_mid = abs(williams_r_aligned[i] + 50) < 5  # Near -50
        
        # Volume confirmation: current 6h volume > 2.0x 20-period average
        volume_spike = volume_6h_aligned[i] > 2.0 * vol_ma_20_aligned[i]
        
        # Entry conditions
        if bullish_trend and oversold and volume_spike and position != 1:
            position = 1
            signals[i] = position_size
        elif bearish_trend and overbought and volume_spike and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions: Williams %R reverts to midpoint or opposite extreme
        elif (revert_to_mid or 
              (bullish_trend and williams_r_aligned[i] > -20) or 
              (bearish_trend and williams_r_aligned[i] < -80)) and position != 0:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_williams_r_mean_reversion_trend_volume_v2"
timeframe = "6h"
leverage = 1.0