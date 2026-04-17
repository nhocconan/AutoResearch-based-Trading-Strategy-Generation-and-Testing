# 1d_WeeklyDonchianBreakout_Volume_Trend
# Hypothesis: 1d Donchian(20) breakout with weekly EMA trend filter and volume confirmation.
# Weekly trend filter ensures we trade with the dominant trend (works in bull via long breakouts, bear via short breakdowns).
# Volume confirmation reduces false breakouts. Target: 15-25 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian and EMA
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    high_max_20 = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly EMA50 for trend filter
    close_weekly_series = pd.Series(close_weekly)
    ema50_weekly = close_weekly_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly Donchian and EMA to daily
    donchian_high_daily = align_htf_to_ltf(prices, df_weekly, high_max_20)
    donchian_low_daily = align_htf_to_ltf(prices, df_weekly, low_min_20)
    ema50_daily = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Volume filter: current volume > 1.5 * 20-period average (moderate to balance signal quality)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 70  # Need 20-period weekly Donchian + EMA50 + volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_daily[i]) or 
            np.isnan(donchian_low_daily[i]) or 
            np.isnan(ema50_daily[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.5x average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below weekly EMA50
        price_above_ema = close[i] > ema50_daily[i]
        price_below_ema = close[i] < ema50_daily[i]
        
        # Price relative to weekly Donchian channels
        price_above_high = close[i] > donchian_high_daily[i]
        price_below_low = close[i] < donchian_low_daily[i]
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high with volume and above weekly EMA50
            if (price_above_high and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian low with volume and below weekly EMA50
            elif (price_below_low and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below weekly Donchian low OR below weekly EMA50
            if (close[i] < donchian_low_daily[i]) or (close[i] < ema50_daily[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above weekly Donchian high OR above weekly EMA50
            if (close[i] > donchian_high_daily[i]) or (close[i] > ema50_daily[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchianBreakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0