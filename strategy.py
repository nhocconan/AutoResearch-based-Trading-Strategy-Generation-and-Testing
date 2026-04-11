#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Uses weekly timeframe for primary trend (bull/bear regime)
# - Daily Camarilla levels (R3/S3 for fade, R4/S4 for breakout) on 6h chart
# - Volume confirmation to filter weak breakouts
# - Designed to work in both bull (breakouts with volume) and bear (breakdowns with volume) markets
# - Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag

name = "6h_1w_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute weekly EMA trend filter (34/89 for responsiveness)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1w = pd.Series(close_1w).ewm(span=89, adjust=False, min_periods=89).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    ema_89_aligned = align_htf_to_ltf(prices, df_1w, ema_89_1w)
    
    # Weekly trend bias (1 for uptrend, -1 for downtrend)
    trend_bias = np.zeros(len(ema_34_aligned))
    trend_bias[ema_34_aligned > ema_89_aligned] = 1
    trend_bias[ema_34_aligned < ema_89_aligned] = -1
    
    # Load daily data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily range
    daily_range = high_1d - low_1d
    
    # Camarilla levels (based on previous day's OHLC)
    camarilla_r4 = close_1d + (1.1/2.0) * daily_range  # R4 = Close + 1.1*(High-Low)/2
    camarilla_r3 = close_1d + (1.1/4.0) * daily_range  # R3 = Close + 1.1*(High-Low)/4
    camarilla_s3 = close_1d - (1.1/4.0) * daily_range  # S3 = Close - 1.1*(High-Low)/4
    camarilla_s4 = close_1d - (1.1/2.0) * daily_range  # S4 = Close - 1.1*(High-Low)/2
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Pre-compute 6h volume confirmation (20-period average)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(trend_bias[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        volume_current = volume[i]
        
        # Trend filter from weekly
        trend_up = trend_bias[i] == 1
        trend_down = trend_bias[i] == -1
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Camarilla breakout/fade conditions
        # Breakout long: price closes above R4 with volume and weekly uptrend
        breakout_long = price_close > r4_aligned[i-1] and trend_up and vol_confirm
        
        # Breakout short: price closes below S4 with volume and weekly downtrend
        breakout_short = price_close < s4_aligned[i-1] and trend_down and vol_confirm
        
        # Fade long: price touches S3 and reverses up in weekly downtrend (mean reversion in bear)
        fade_long = (price_close <= s3_aligned[i-1] * 1.002) and trend_down and vol_confirm
        
        # Fade short: price touches R3 and reverses down in weekly uptrend (mean reversion in bull)
        fade_short = (price_close >= r3_aligned[i-1] * 0.998) and trend_up and vol_confirm
        
        # Entry conditions
        enter_long = breakout_long or fade_long
        enter_short = breakout_short or fade_short
        
        # Exit conditions: opposite Camarilla level or trend reversal
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price reaches S3 (for breakout) or R3 (for fade) or trend turns down
            exit_long = (price_close < s3_aligned[i-1]) or (price_close > r3_aligned[i-1]) or (not trend_up)
        elif position == -1:
            # Exit short if price reaches R3 (for breakout) or S3 (for fade) or trend turns up
            exit_short = (price_close > r3_aligned[i-1]) or (price_close < s3_aligned[i-1]) or (not trend_down)
        
        # Trading logic with discrete position sizing (±0.25)
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals