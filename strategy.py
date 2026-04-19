#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with weekly trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) AND price > weekly EMA200 (bullish trend) AND volume > 1.3x daily average volume
# Short when Williams %R > -20 (overbought) AND price < weekly EMA200 (bearish trend) AND volume > 1.3x daily average volume
# Exit when Williams %R returns to -50 (mean reversion) or trend weakens
# Williams %R identifies overbought/oversold conditions, weekly EMA200 filters trend direction, volume confirms momentum.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years).
name = "6h_WilliamsR_MeanReversion_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams %R calculation (14-period)
    df_1d = get_htf_data(prices, '1d')
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_1d['close'].values) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Align Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get weekly data for trend filter (EMA 200)
    df_1w = get_htf_data(prices, '1w')
    weekly_ema200 = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    weekly_ema200_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema200)
    
    # Get daily average volume for confirmation (20-period)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 200)  # Ensure Williams %R and weekly EMA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(weekly_ema200_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        wr = williams_r_aligned[i]
        weekly_ema = weekly_ema200_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        
        # Williams %R levels
        oversold = wr < -80
        overbought = wr > -20
        mean_reversion = abs(wr + 50) < 10  # near -50 (mean level)
        
        # Weekly trend filter
        bullish_trend = price > weekly_ema
        bearish_trend = price < weekly_ema
        
        # Volume confirmation
        volume_confirmed = vol > 1.3 * vol_ma
        
        if position == 0:
            # Long entry: oversold + bullish trend + volume confirmation
            if oversold and bullish_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: overbought + bearish trend + volume confirmation
            elif overbought and bearish_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: mean reversion OR trend weakens
            if mean_reversion or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: mean reversion OR trend weakens
            if mean_reversion or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals