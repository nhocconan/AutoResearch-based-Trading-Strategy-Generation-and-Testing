#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with weekly Donchian breakout + volume confirmation + weekly ATR filter
# - Uses weekly Donchian channels (20-period) for trend-following breakouts
# - Volume confirmation ensures breakouts have institutional participation
# - Weekly ATR filter adapts to volatility regimes, avoiding low-volatility false breakouts
# - Designed to work in both bull (breakouts continue) and bear (breakouts fail quickly) markets
# - Target: 20-50 trades over 4 years (5-12/year) to minimize fee drag while capturing major moves
# - Discrete position sizing (0.25) to reduce churn and manage drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # 20-period ATR for volatility measurement (weekly)
    high_low = high_1w - low_1w
    high_close = np.abs(high_1w - np.roll(close_1w, 1))
    low_close = np.abs(low_1w - np.roll(close_1w, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    tr_series = pd.Series(tr)
    atr_20w = tr_series.rolling(window=20, min_periods=20).mean().values
    
    # Weekly volume filter: current volume > 1.8x 20-period average
    vol_series = pd.Series(volume_1w)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Weekly Donchian channels (20-period) - breakout levels
    high_series = pd.Series(high_1w)
    low_series = pd.Series(low_1w)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate weekly volatility as ATR normalized by price
    weekly_volatility = atr_20w / close_1w
    weekly_vol_series = pd.Series(weekly_volatility)
    # Use 70th percentile of weekly volatility over 8 weeks as threshold
    vol_threshold = weekly_vol_series.rolling(window=8, min_periods=8).quantile(0.70).values
    # Align volatility threshold to daily timeframe
    vol_threshold_1d = align_htf_to_ltf(prices, df_1w, vol_threshold)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if np.isnan(atr_20w[i-1]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(vol_threshold_1d[i]):
            continue
        
        # Get previous week's data for volatility-based levels
        prev_close = close_1w[i-1]
        prev_atr = atr_20w[i-1]  # Previous week's ATR
        
        # Calculate volatility-adjusted threshold (0.4 * ATR) - balanced for signal frequency
        threshold = prev_atr * 0.4
        
        # Calculate dynamic breakout levels based on volatility
        upper_break = prev_close + threshold
        lower_break = prev_close - threshold
        
        # Create arrays for alignment
        upper_array = np.full(len(df_1w), upper_break)
        lower_array = np.full(len(df_1w), lower_break)
        
        upper_break_1d = align_htf_to_ltf(prices, df_1w, upper_array)[i]
        lower_break_1d = align_htf_to_ltf(prices, df_1w, lower_array)[i]
        
        if position == 0:
            # Long: Price breaks above upper level with volume and high volatility regime
            if (close[i] > upper_break_1d and close[i-1] <= upper_break_1d and 
                volume[i] > vol_ma[i] * 1.8 and 
                weekly_volatility[i] > vol_threshold_1d[i]):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below lower level with volume and high volatility regime
            elif (close[i] < lower_break_1d and close[i-1] >= lower_break_1d and 
                  volume[i] > vol_ma[i] * 1.8 and 
                  weekly_volatility[i] > vol_threshold_1d[i]):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price breaks below lower level (reversal) or drops below weekly Donchian low
            if close[i] < lower_break_1d or close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price breaks above upper level (reversal) or rises above weekly Donchian high
            if close[i] > upper_break_1d or close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_1w_VolatilityAdjusted_Donchian_Breakout_Volume_v1"
timeframe = "1d"
leverage = 1.0