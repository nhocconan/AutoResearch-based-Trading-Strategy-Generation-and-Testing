#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Williams %R (Williams %R) with overbought/oversold levels and mean reversion
# - Uses Williams %R (14) calculated on weekly timeframe for overbought/oversold signals
# - Long when weekly Williams %R crosses above -80 from below (oversold bounce)
# - Short when weekly Williams %R crosses below -20 from above (overbought rejection)
# - Filters trades with daily RSI(14) to avoid counter-trend entries in strong trends
# - Uses volume confirmation (volume > 1.5x 20-period MA) to ensure participation
# - Designed to work in ranging markets (mean reversion at extremes) and avoid strong trends
# - Target: 30-100 total trades over 4 years (7-25/year) with 0.25 position sizing

name = "1d_WilliamsR_Weekly_OBOS_MeanReversion"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Williams %R calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R on weekly data: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Williams %R = -100 * (HH - C) / (HH - LL), ranges from 0 to -100
    # Overbought: > -20, Oversold: < -80
    highest_high = pd.Series(df_weekly['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_weekly['low']).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_ww = highest_high - lowest_low
    range_ww = np.where(range_ww == 0, 0.0001, range_ww)
    
    williams_r = -100 * (highest_high - df_weekly['close'].values) / range_ww
    
    # Align Williams %R to daily timeframe (no extra delay needed as it's based on current weekly bar)
    williams_r_daily = align_htf_to_ltf(prices, df_weekly, williams_r)
    
    # Daily RSI(14) for trend filter (avoid counter-trend in strong trends)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 0.0001, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.5x 20-day moving average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup for indicators
        # Skip if any critical value is NaN
        if (np.isnan(williams_r_daily[i]) or np.isnan(rsi[i]) or np.isnan(volume_confirmation[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long signal: Williams %R crosses above -80 from below (oversold bounce)
            williams_r_prev = williams_r_daily[i-1] if i > 0 else williams_r_daily[i]
            if (williams_r_prev <= -80 and williams_r_daily[i] > -80 and 
                rsi[i] < 60 and volume_confirmation[i]):  # Avoid overbought RSI
                signals[i] = 0.25
                position = 1
            # Short signal: Williams %R crosses below -20 from above (overbought rejection)
            elif (williams_r_prev >= -20 and williams_r_daily[i] < -20 and 
                  rsi[i] > 40 and volume_confirmation[i]):  # Avoid oversold RSI
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R reaches overbought (-20) or RSI overbought
            if williams_r_daily[i] >= -20 or rsi[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R reaches oversold (-80) or RSI oversold
            if williams_r_daily[i] <= -80 or rsi[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals