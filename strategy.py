#!/usr/bin/env python3
name = "1d_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels from previous day
    df_1d_high = prices['high'].rolling(2).apply(lambda x: x[0], raw=True).shift(1)  # previous day high
    df_1d_low = prices['low'].rolling(2).apply(lambda x: x[0], raw=True).shift(1)    # previous day low
    df_1d_close = prices['close'].rolling(2).apply(lambda x: x[0], raw=True).shift(1) # previous day close
    
    # Calculate R3 and S3 levels
    r3 = df_1d_close + 1.1 * (df_1d_high - df_1d_low) / 4
    s3 = df_1d_close - 1.1 * (df_1d_high - df_1d_low) / 4
    
    # 1w EMA34 trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 1  # 1 day cooldown to reduce trades
    
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3[i]) or 
            np.isnan(s3[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 1w trend direction
        trend_up = close > ema_34_1w_aligned[i]
        trend_down = close < ema_34_1w_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Break above R3 in uptrend with strong volume
            if (close[i] > r3[i] and 
                trend_up[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Break below S3 in downtrend with strong volume
            elif (close[i] < s3[i] and 
                  trend_down[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price re-enters Camarilla body (between R3 and S3) or trend change
            if (close[i] < r3[i] and close[i] > s3[i]) or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price re-enters Camarilla body or trend change
            if (close[i] < r3[i] and close[i] > s3[i]) or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Using daily timeframe with Camarilla R3/S3 breakouts, 1-week EMA34 trend filter, and 2.0x volume spike
# will yield 10-25 trades per year (40-100 total over 4 years), minimizing fee drag. The strategy trades
# with the higher timeframe trend, capturing institutional breakouts in both bull and bear markets.
# Position size of 0.25 manages drawdown, and daily cooldown prevents overtrading. Focus on BTC/ETH as primary targets.