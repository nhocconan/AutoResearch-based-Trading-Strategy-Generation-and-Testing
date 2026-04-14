# I have read and understood all the rules.
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily price channel breakout with weekly trend filter and volume confirmation
# Long when price breaks above daily Donchian upper band with volume spike and weekly bullish trend
# Short when price breaks below daily Donchian lower band with volume spike and weekly bearish trend
# Exit when price crosses the Donchian midline
# Uses weekly EMA trend filter to avoid counter-trend trades and reduce whipsaw
# Target: 30-100 trades per symbol over 4 years (7-25/year) to minimize fee drag
# This strategy focuses on the daily timeframe to reduce noise and improve signal quality
# The weekly trend filter ensures alignment with higher timeframe momentum
# Volume confirmation ensures breakouts have institutional participation

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
    
    # Calculate daily Donchian channel (20-period lookback)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    donchian_upper = pd.Series(high_daily).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_daily).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate weekly EMA for trend filter (21-period)
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate daily volume average (20-period)
    vol_daily = df_daily['volume'].values
    vol_ma_daily = pd.Series(vol_daily).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to daily timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_daily, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_daily, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_daily, donchian_middle)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    vol_ma_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_daily)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 40  # for 20-period calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_weekly_aligned[i]) or np.isnan(vol_ma_daily_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_daily_current = volume[i]  # Current daily volume
        
        if position == 0:
            # Long setup: break above Donchian upper with volume spike and weekly bullish trend
            if (price > donchian_upper_aligned[i] and 
                vol_daily_current > 1.8 * vol_ma_daily_aligned[i] and  # Volume spike
                price > ema_weekly_aligned[i]):                    # Price above weekly EMA for bullish trend
                position = 1
                signals[i] = position_size
            # Short setup: break below Donchian lower with volume spike and weekly bearish trend
            elif (price < donchian_lower_aligned[i] and 
                  vol_daily_current > 1.8 * vol_ma_daily_aligned[i] and  # Volume spike
                  price < ema_weekly_aligned[i]):                    # Price below weekly EMA for bearish trend
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian middle
            if price < donchian_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian middle
            if price > donchian_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Donchian_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0