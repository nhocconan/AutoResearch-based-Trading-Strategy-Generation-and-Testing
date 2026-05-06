#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above 4h Camarilla R3 AND 1d EMA34 > EMA89 AND volume > 1.8 * avg_volume(20)
# Short when price breaks below 4h Camarilla S3 AND 1d EMA34 < EMA89 AND volume > 1.8 * avg_volume(20)
# Exit when price crosses 4h EMA34 (trend reversal signal)
# Uses discrete sizing 0.20 to minimize fee impact and control drawdown
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# 4h Camarilla provides intermediate structure with clear breakout/fade levels
# 1d EMA34/EMA89 filter ensures alignment with higher timeframe trend
# Volume confirmation filters weak breakouts
# Session filter (08-20 UTC) reduces noise during low-liquidity periods
# Works in bull (breakouts above R3 in uptrend) and bear (breakdowns below S3 in downtrend)

name = "1h_4hCamarillaR3S3_1dEMA34Trend_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data ONCE before loop for Camarilla pivots
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need sufficient data for Camarilla calculation
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla levels (based on previous 4h bar)
    typical_price_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    camarilla_r3_4h = typical_price_4h + 0.275 * range_4h
    camarilla_s3_4h = typical_price_4h - 0.275 * range_4h
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 89:  # Need sufficient data for EMA89
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 and EMA89 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema_34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1d = close_series_1d.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 4h Camarilla levels to 1h timeframe (wait for completed 4h bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    
    # Align 1d EMA indicators to 1h timeframe (wait for completed 1d bar)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_89_aligned = align_htf_to_ltf(prices, df_1d, ema_89_1d)
    
    # Calculate volume confirmation: volume > 1.8 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(ema_89_aligned[i]) or np.isnan(avg_volume_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Camarilla R3 with 1d EMA34 > EMA89 and volume confirmation
            if (close[i] > camarilla_r3_aligned[i] and close[i-1] <= camarilla_r3_aligned[i-1] and 
                ema_34_aligned[i] > ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Camarilla S3 with 1d EMA34 < EMA89 and volume confirmation
            elif (close[i] < camarilla_s3_aligned[i] and close[i-1] >= camarilla_s3_aligned[i-1] and 
                  ema_34_aligned[i] < ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h EMA34 (trend reversal)
            # Note: We use 4h EMA34 for exit to match the timeframe of entry signal
            df_4h_close = get_htf_data(prices, '4h')['close'].values
            ema_34_4h = pd.Series(df_4h_close).ewm(span=34, adjust=False, min_periods=34).mean().values
            ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
            if close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above 4h EMA34 (trend reversal)
            df_4h_close = get_htf_data(prices, '4h')['close'].values
            ema_34_4h = pd.Series(df_4h_close).ewm(span=34, adjust=False, min_periods=34).mean().values
            ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
            if close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals