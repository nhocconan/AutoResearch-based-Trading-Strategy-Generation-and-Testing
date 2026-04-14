#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams %R with daily trend filter and volume confirmation
# Long when Williams %R crosses above -20 (exit oversold) with daily bullish trend and volume spike
# Short when Williams %R crosses below -80 (exit overbought) with daily bearish trend and volume spike
# Exit when Williams %R crosses back to opposite extreme or center (-50)
# Uses daily EMA trend filter to avoid counter-trend trades
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost
# Williams %R is effective at catching reversals in ranging markets and pullbacks in trends

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate 6-hour Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate daily EMA for trend filter (21-period)
    close_daily = df_daily['close'].values
    ema_daily = pd.Series(close_daily).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 6-hour volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_daily, williams_r)
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    vol_ma_aligned = align_htf_to_ltf(prices, df_daily, vol_ma)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # for 20-period volume MA and 14-period Williams %R
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_daily_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        wr_prev = williams_r_aligned[i-1]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long setup: Williams %R crosses above -20 (exiting oversold) with volume spike and daily bullish trend
            if (wr > -20 and wr_prev <= -20 and  # Cross above -20
                vol_current > 1.5 * vol_ma_aligned[i] and  # Volume spike
                price > ema_daily_aligned[i]):             # Price above daily EMA for bullish trend
                position = 1
                signals[i] = position_size
            # Short setup: Williams %R crosses below -80 (exiting overbought) with volume spike and daily bearish trend
            elif (wr < -80 and wr_prev >= -80 and    # Cross below -80
                  vol_current > 1.5 * vol_ma_aligned[i] and  # Volume spike
                  price < ema_daily_aligned[i]):             # Price below daily EMA for bearish trend
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R crosses below -50 (momentum fading) or above -10 (overbought)
            if wr < -50 or wr > -10:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R crosses above -50 (momentum fading) or below -90 (oversold)
            if wr > -50 or wr < -90:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_WilliamsR_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0