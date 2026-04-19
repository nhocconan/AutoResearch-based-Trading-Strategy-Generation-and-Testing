#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Bollinger Bands and RSI for mean reversion
# - Weekly Bollinger Bands (20, 2) identify overbought/oversold conditions
# - Daily RSI(14) provides entry timing: long when RSI < 30 near lower BB, short when RSI > 70 near upper BB
# - Weekly trend filter: only take longs when price > weekly SMA(50), shorts when price < weekly SMA(50)
# - Volume confirmation: daily volume > 1.5x 20-day average for conviction
# - Designed to work in ranging markets (mean reversion) and trending markets (with trend filter)
# - Target: 10-20 trades/year to minimize fee drag

name = "1d_WeeklyBB_RSI_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands and trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly Bollinger Bands (20, 2)
    weekly_close = df_weekly['close'].values
    weekly_ma = pd.Series(weekly_close).rolling(window=20, min_periods=20).mean().values
    weekly_std = pd.Series(weekly_close).rolling(window=20, min_periods=20).std().values
    upper_band = weekly_ma + 2 * weekly_std
    lower_band = weekly_ma - 2 * weekly_std
    
    # Align weekly bands to daily
    upper_band_aligned = align_htf_to_ltf(prices, df_weekly, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_weekly, lower_band)
    weekly_ma_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ma)
    
    # Weekly SMA(50) for trend filter
    weekly_sma50 = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    weekly_sma50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_sma50)
    
    # Daily RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Daily volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(weekly_ma_aligned[i]) or np.isnan(weekly_sma50_aligned[i]) or 
            np.isnan(rsi_values[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-day average
        volume_filter = vol_ma[i] > 0 and volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for long entry: price near lower BB, RSI oversold, uptrend filter
            if (close[i] <= lower_band_aligned[i] * 1.02 and  # Within 2% of lower band
                rsi_values[i] < 30 and 
                close[i] > weekly_sma50_aligned[i] and  # Uptrend filter
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Look for short entry: price near upper BB, RSI overbought, downtrend filter
            elif (close[i] >= upper_band_aligned[i] * 0.98 and  # Within 2% of upper band
                  rsi_values[i] > 70 and 
                  close[i] < weekly_sma50_aligned[i] and  # Downtrend filter
                  volume_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price reaches weekly mean or RSI overbought
            if close[i] >= weekly_ma_aligned[i] or rsi_values[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price reaches weekly mean or RSI oversold
            if close[i] <= weekly_ma_aligned[i] or rsi_values[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals