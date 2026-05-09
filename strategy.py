#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion with volume confirmation and weekly trend filter.
# Williams %R identifies overbought/oversold conditions on daily timeframe.
# Enter long when %R < -80 (oversold) and short when %R > -20 (overbought).
# Volume > 1.5x 20-period EMA confirms participation.
# Weekly EMA200 filter ensures trades align with higher timeframe trend.
# Designed for mean reversion in ranging markets with trend filter to avoid counter-trend trades.
name = "1d_WilliamsR_MeanReversion_Volume_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for EMA200 trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 200:
        return np.zeros(n)
    
    # Daily data for Williams %R calculation (14-period)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_daily).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_daily).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_daily) / (highest_high - lowest_low) * -100
    
    # Use previous day's Williams %R (shift by 1 to avoid look-ahead)
    williams_r_shifted = np.roll(williams_r, 1)
    williams_r_shifted[0] = np.nan
    
    # Align Williams %R to daily timeframe (no additional shift needed as we already used previous day)
    williams_r_daily = align_htf_to_ltf(prices, df_daily, williams_r_shifted)
    
    # Weekly EMA200 trend filter
    ema_200_weekly = pd.Series(df_weekly['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_200_weekly)
    
    # Volume spike filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for weekly EMA200 to be ready
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(williams_r_daily[i]) or np.isnan(ema_200_weekly_aligned[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) with volume spike and price above weekly EMA200
            if (williams_r_daily[i] < -80 and vol_spike[i] and price > ema_200_weekly_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) with volume spike and price below weekly EMA200
            elif (williams_r_daily[i] > -20 and vol_spike[i] and price < ema_200_weekly_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns above -50 (mean reversion) or weekly trend turns bearish
            if williams_r_daily[i] > -50 or price < ema_200_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns below -50 (mean reversion) or weekly trend turns bullish
            if williams_r_daily[i] < -50 or price > ema_200_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals