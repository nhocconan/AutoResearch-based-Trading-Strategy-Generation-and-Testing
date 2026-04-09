#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d HTF for signal direction and 1h for entry timing.
# - 4h trend: EMA50 > EMA200 = uptrend, < = downtrend
# - 1d regime: Choppiness Index (CHOP) > 61.8 = range (mean revert), < 38.2 = trending (trend follow)
# - 1h entry: In uptrend + trending regime, buy pullback to 20 EMA; in downtrend + trending regime, sell pullback to 20 EMA
# - In range regime (CHOP between 38.2 and 61.8), fade extremes: buy near 20-period low, sell near 20-period high
# - Volume confirmation: current 1h volume > 1.5x 20-period average
# - Session filter: 08-20 UTC only
# - Fixed position size 0.20 to control drawdown and minimize fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years)

name = "1h_4h_1d_chop_trend_meanrev_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMAs for trend filter
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 4h EMAs to 1h timeframe (wait for completed 4h bar)
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Choppiness Index (CHOP)
    def calculate_chop(high, low, close, window=14):
        atr = np.zeros_like(close)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        
        hh = pd.Series(high).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low).rolling(window=window, min_periods=window).min().values
        
        chop = np.zeros_like(close)
        for i in range(len(close)):
            if atr[i] > 0 and hh[i] > ll[i]:
                log_sum = np.log(atr[i] * window / (hh[i] - ll[i])) / np.log(2)
                chop[i] = 100 * log_sum / np.log(window)
            else:
                chop[i] = 50.0  # Neutral when invalid
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, window=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Pre-compute 1h indicators for entry timing
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    ema_20_1h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    position_size = 0.20
    
    for i in range(100, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_4h_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(ema_20_1h[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(highest_high_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # 4h trend filter
        uptrend_4h = ema_50_4h_aligned[i] > ema_200_4h_aligned[i]
        downtrend_4h = ema_50_4h_aligned[i] < ema_200_4h_aligned[i]
        
        # 1d regime filter
        chop_val = chop_1d_aligned[i]
        trending_regime = chop_val < 38.2
        range_regime = chop_val > 61.8
        
        if position == 1:  # Long position
            # Exit when price closes below 1h 20 EMA (trend/momentum loss)
            if close[i] < ema_20_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when price closes above 1h 20 EMA (trend/momentum loss)
            if close[i] > ema_20_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            if volume_confirmed:
                # Trending regime: trade with 4h trend using 1h pullbacks
                if trending_regime:
                    # Long: uptrend + pullback to 20 EMA
                    if uptrend_4h and close[i] <= ema_20_1h[i] * 1.001:  # Allow small overshoot
                        position = 1
                        signals[i] = position_size
                    # Short: downtrend + pullback to 20 EMA
                    elif downtrend_4h and close[i] >= ema_20_1h[i] * 0.999:  # Allow small overshoot
                        position = -1
                        signals[i] = -position_size
                # Range regime: mean reversion at extremes
                elif range_regime:
                    # Long: near 20-period low
                    if low[i] <= lowest_low_20[i] * 1.002:  # Allow small overshoot
                        position = 1
                        signals[i] = position_size
                    # Short: near 20-period high
                    elif high[i] >= highest_high_20[i] * 0.998:  # Allow small overshoot
                        position = -1
                        signals[i] = -position_size
    
    return signals