# Hypothesis: 12h strategy using daily pivot points with volume confirmation and 1w trend filter
# Uses contrarian entries at S1/R1 in strong trends and breakouts at R2/S2 in weak trends
# Designed for low trade frequency (<30/year) to avoid fee drag, works in bull/bear via trend adaptation
# Weekly trend filter avoids counter-trend trades in strong moves, reducing whipsaw

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    
    # Get weekly data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily pivot levels to 12h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Align weekly EMA50 to 12h
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    # Calculate RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Get current weekly trend
        price_above_ema = close[i] > ema50_1w_aligned[i]
        price_below_ema = close[i] < ema50_1w_aligned[i]
        
        # Strong trend: price > 2% away from EMA50
        strong_uptrend = price_above_ema and (close[i] > ema50_1w_aligned[i] * 1.02)
        strong_downtrend = price_below_ema and (close[i] < ema50_1w_aligned[i] * 0.98)
        
        # Weak trend/ranging: price within 1% of EMA50
        weak_trend = abs(close[i] - ema50_1w_aligned[i]) < (ema50_1w_aligned[i] * 0.01)
        
        # Fade conditions at S1/R1 (counter-trend in weak markets)
        fade_s1 = close[i] <= s1_aligned[i] and rsi[i] < 35 and volume_confirm[i]
        fade_r1 = close[i] >= r1_aligned[i] and rsi[i] > 65 and volume_confirm[i]
        
        # Breakout conditions at S2/R2 (trend continuation)
        breakout_s2 = close[i] < s2_aligned[i] and rsi[i] < 40 and volume_confirm[i]
        breakout_r2 = close[i] > r2_aligned[i] and rsi[i] > 60 and volume_confirm[i]
        
        # Long logic
        long_signal = False
        if weak_trend:
            # In ranging markets, fade at S1
            long_signal = fade_s1
        elif strong_uptrend:
            # In strong uptrend, buy dips to S1
            long_signal = fade_s1
        else:
            # In strong downtrend or transition, look for S2 breakouts
            long_signal = breakout_s2
        
        # Short logic
        short_signal = False
        if weak_trend:
            # In ranging markets, fade at R1
            short_signal = fade_r1
        elif strong_downtrend:
            # In strong downtrend, sell rallies to R1
            short_signal = fade_r1
        else:
            # In strong uptrend or transition, look for R2 breakouts
            short_signal = breakout_r2
        
        # Execute signals with position management
        if long_signal and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_signal and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit on opposite signal or extreme RSI reversal
        elif position == 1 and (rsi[i] > 75 or short_signal):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (rsi[i] < 25 or long_signal):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_DailyPivot_WeeklyTrend_FadeBreakout"
timeframe = "12h"
leverage = 1.0