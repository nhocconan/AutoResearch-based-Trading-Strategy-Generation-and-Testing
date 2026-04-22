#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d mean-reversion at weekly Bollinger Band extremes with volume confirmation
# Works in bull (buying dips in uptrend) and bear (selling rallies in downtrend)
# Target: 10-20 trades/year, low frequency avoids fee drag
# Uses weekly trend filter to avoid counter-trend whipsaws

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend
    weekly_close = df_weekly['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema50)
    
    # Load daily data for Bollinger Bands
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Daily Bollinger Bands (20, 2.0)
    daily_close = df_daily['close'].values
    daily_ma20 = pd.Series(daily_close).rolling(window=20, min_periods=20).mean().values
    daily_std20 = pd.Series(daily_close).rolling(window=20, min_periods=20).std().values
    upper_band = daily_ma20 + 2.0 * daily_std20
    lower_band = daily_ma20 - 2.0 * daily_std20
    
    # Align Bollinger Bands to 1d timeframe (already aligned but keep for consistency)
    upper_band_aligned = align_htf_to_ltf(prices, df_daily, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_daily, lower_band)
    
    # Daily volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(weekly_ema50_aligned[i]) or np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price touches lower Bollinger Band with volume, in weekly uptrend
            if (close[i] <= lower_band_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20[i] and
                weekly_ema50_aligned[i] > weekly_ema50_aligned[max(0, i-5)]):  # Rising weekly trend
                signals[i] = 0.25
                position = 1
            # Short: Price touches upper Bollinger Band with volume, in weekly downtrend
            elif (close[i] >= upper_band_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i] and
                  weekly_ema50_aligned[i] < weekly_ema50_aligned[max(0, i-5)]):  # Falling weekly trend
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to middle Bollinger Band (mean reversion complete)
            if position == 1:
                if close[i] >= daily_ma20[min(i, len(daily_ma20)-1)]:  # Use current daily MA20
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] <= daily_ma20[min(i, len(daily_ma20)-1)]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_BollingerMeanReversion_WeeklyTrend"
timeframe = "1d"
leverage = 1.0