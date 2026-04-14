#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily price action with weekly trend filter and volume confirmation
# Long when price breaks above daily resistance with volume spike and weekly bullish trend
# Short when price breaks below daily support with volume spike and weekly bearish trend
# Exit on daily midline cross
# Uses weekly trend to avoid counter-trend trades in bear markets
# Target: 30-100 trades total over 4 years (7-25/year)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily and weekly data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate daily support/resistance levels (20-period lookback)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    resistance = pd.Series(high_daily).rolling(window=20, min_periods=20).max().values
    support = pd.Series(low_daily).rolling(window=20, min_periods=20).min().values
    midline = (resistance + support) / 2
    
    # Calculate weekly EMA for trend filter (21-period)
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate daily volume average (20-period)
    vol_daily = df_daily['volume'].values
    vol_ma_daily = pd.Series(vol_daily).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to daily timeframe
    resistance_aligned = align_htf_to_ltf(prices, df_daily, resistance)
    support_aligned = align_htf_to_ltf(prices, df_daily, support)
    midline_aligned = align_htf_to_ltf(prices, df_daily, midline)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    vol_ma_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_daily)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 60  # for 20-period calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(resistance_aligned[i]) or np.isnan(support_aligned[i]) or 
            np.isnan(ema_weekly_aligned[i]) or np.isnan(vol_ma_daily_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_daily_current = volume[i]  # Current daily volume
        
        if position == 0:
            # Long setup: break above resistance with volume spike and weekly bullish trend
            if (price > resistance_aligned[i] and 
                vol_daily_current > 1.8 * vol_ma_daily_aligned[i] and  # Volume spike
                price > ema_weekly_aligned[i]):                    # Price above weekly EMA for bullish trend
                position = 1
                signals[i] = position_size
            # Short setup: break below support with volume spike and weekly bearish trend
            elif (price < support_aligned[i] and 
                  vol_daily_current > 1.8 * vol_ma_daily_aligned[i] and  # Volume spike
                  price < ema_weekly_aligned[i]):                    # Price below weekly EMA for bearish trend
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below midline
            if price < midline_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above midline
            if price > midline_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Resistance_Support_Volume_WeeklyTrend"
timeframe = "1d"
leverage = 1.0