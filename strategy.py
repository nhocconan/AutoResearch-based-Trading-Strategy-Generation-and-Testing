# %%
#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_Breakout_1wTrend_TradingSession"
timeframe = "4h"
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
    
    # Weekly OHLC for trend (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA20 trend
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    trend_up = close > ema_20_1w_aligned
    trend_down = close < ema_20_1w_aligned
    
    # Daily OHLC for Camarilla R3/S3 levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Camarilla R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3 = daily_close + (daily_high - daily_low) * 1.1 / 4
    camarilla_s3 = daily_close - (daily_high - daily_low) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume filter: current volume > 1.8x 28-period average
    vol_ma_28 = np.full(n, np.nan)
    for i in range(28, n):
        vol_ma_28[i] = np.mean(volume[i-28:i])
    vol_filter = volume > (1.8 * vol_ma_28)
    
    # Trading session filter: UTC 8:00-20:00 (most liquid session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 12  # ~2 days (12*4h) to prevent overtrading
    
    start_idx = max(28, 30)  # Ensure enough data for volume MA and stability
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction
        trending_up = trend_up[i]
        trending_down = trend_down[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above Camarilla R3 with volume in weekly uptrend during liquid session
            if (close[i] > r3_aligned[i] and 
                trending_up and 
                vol_filter[i] and 
                session_filter[i]):
                signals[i] = 0.30
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below Camarilla S3 with volume in weekly downtrend during liquid session
            elif (close[i] < s3_aligned[i] and 
                  trending_down and 
                  vol_filter[i] and 
                  session_filter[i]):
                signals[i] = -0.30
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls back below Camarilla S3 or weekly trend changes to down
            if close[i] < s3_aligned[i] or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: Price rises back above Camarilla R3 or weekly trend changes to up
            if close[i] > r3_aligned[i] or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: On 4h timeframe, price breaking above/below Camarilla R3/S3 levels with volume confirmation, weekly EMA20 trend filter, and liquid trading session (UTC 8-20) captures institutional breakout momentum. Camarilla R3/S3 represent stronger support/resistance than R1/S1, reducing false breakouts. Weekly trend filter ensures alignment with higher timeframe momentum. Session filter avoids low-liquidity periods. Target: 50-120 trades over 4 years (12-30/year) to minimize fee drag while capturing significant moves. Works in bull markets (breakouts above R3 in weekly uptrend) and bear markets (breakdowns below S3 in weekly downtrend).