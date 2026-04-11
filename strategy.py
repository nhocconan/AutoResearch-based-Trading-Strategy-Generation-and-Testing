#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-week and 1-day Bollinger Bands combined with 
# weekly trend filter (KAMA) and daily volume confirmation. Uses BB width for 
# regime detection - narrow bands for mean reversion at ±2σ, wide bands for 
# breakout continuation. Designed for low-frequency, high-conviction trades 
# (target: 15-35 trades/year) to minimize fee drag while capturing both 
# reversal and momentum moves in bull/bear markets.

name = "12h_1w1d_bb_width_kama_vol"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly and daily data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # === WEEKLY INDICATORS ===
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly KAMA for trend (ER=10, fast=2, slow=30)
    def calculate_kama(close, er_period=10, fast=2, slow=30):
        n = len(close)
        kama = np.full(n, np.nan)
        if n == 0:
            return kama
        
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close, prepend=close[0]))
        
        for i in range(1, n):
            if i >= er_period:
                er_denom = np.sum(volatility[i-er_period+1:i+1])
                if er_denom > 0:
                    er = np.sum(change[i-er_period+1:i+1]) / er_denom
                else:
                    er = 0
                sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
                if i == er_period:
                    kama[i] = close[i-1]
                else:
                    kama[i] = kama[i-1] + sc * (close[i-1] - kama[i-1])
            else:
                kama[i] = close[0]
        return kama
    
    kama_1w = calculate_kama(close_1w)
    weekly_trend_up = close_1w > kama_1w
    weekly_trend_down = close_1w < kama_1w
    
    # Weekly Bollinger Bands (20, 2.0)
    sma_20_1w = np.full_like(close_1w, np.nan)
    std_20_1w = np.full_like(close_1w, np.nan)
    for i in range(19, len(close_1w)):
        sma_20_1w[i] = np.mean(close_1w[i-19:i+1])
        std_20_1w[i] = np.std(close_1w[i-19:i+1])
    
    bb_upper_20_1w = sma_20_1w + 2 * std_20_1w
    bb_lower_20_1w = sma_20_1w - 2 * std_20_1w
    bb_width_20_1w = (bb_upper_20_1w - bb_lower_20_1w) / sma_20_1w * 100  # % width
    
    # === DAILY INDICATORS ===
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Daily Bollinger Bands (20, 2.0)
    sma_20_1d = np.full_like(close_1d, np.nan)
    std_20_1d = np.full_like(close_1d, np.nan)
    for i in range(19, len(close_1d)):
        sma_20_1d[i] = np.mean(close_1d[i-19:i+1])
        std_20_1d[i] = np.std(close_1d[i-19:i+1])
    
    bb_upper_20_1d = sma_20_1d + 2 * std_20_1d
    bb_lower_20_1d = sma_20_1d - 2 * std_20_1d
    bb_width_20_1d = (bb_upper_20_1d - bb_lower_20_1d) / sma_20_1d * 100
    
    # Daily 20-period average volume
    vol_avg_20_1d = np.full_like(volume_1d, np.nan)
    for i in range(19, len(volume_1d)):
        vol_avg_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align all weekly indicators to 12h
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)
    bb_width_20_1w_aligned = align_htf_to_ltf(prices, df_1w, bb_width_20_1w)
    bb_upper_20_1w_aligned = align_htf_to_ltf(prices, df_1w, bb_upper_20_1w)
    bb_lower_20_1w_aligned = align_htf_to_ltf(prices, df_1w, bb_lower_20_1w)
    
    # Align all daily indicators to 12h
    bb_width_20_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_width_20_1d)
    bb_upper_20_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_upper_20_1d)
    bb_lower_20_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_lower_20_1d)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Regime thresholds: narrow band < 3% = mean reversion, wide band > 6% = trend
    NARROW_BB = 3.0
    WIDE_BB = 6.0
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(bb_width_20_1d_aligned[i]) or np.isnan(bb_upper_20_1d_aligned[i]) or
            np.isnan(bb_lower_20_1d_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(weekly_trend_up_aligned[i]) or np.isnan(weekly_trend_down_aligned[i]) or
            np.isnan(bb_width_20_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.8 * daily average volume
        vol_filter = volume[i] > 1.8 * vol_avg_20_1d_aligned[i]
        
        # Determine regime based on weekly BB width
        is_narrow_week = bb_width_20_1w_aligned[i] < NARROW_BB
        is_wide_week = bb_width_20_1w_aligned[i] > WIDE_BB
        
        # Mean reversion signals (narrow weekly bands)
        mr_long = (low[i] <= bb_lower_20_1d_aligned[i] and vol_filter and 
                  weekly_trend_up_aligned[i] and is_narrow_week)
        mr_short = (high[i] >= bb_upper_20_1d_aligned[i] and vol_filter and 
                   weekly_trend_down_aligned[i] and is_narrow_week)
        
        # Breakout/continuation signals (wide weekly bands)
        breakout_long = (high[i] >= bb_upper_20_1d_aligned[i] and vol_filter and
                        weekly_trend_up_aligned[i] and is_wide_week)
        breakout_short = (low[i] <= bb_lower_20_1d_aligned[i] and vol_filter and
                         weekly_trend_down_aligned[i] and is_wide_week)
        
        # Exit conditions
        exit_long = (position == 1 and 
                    (high[i] >= bb_upper_20_1d_aligned[i] or  # Take profit at upper band
                     low[i] <= bb_lower_20_1d_aligned[i]))   # Stop loss at lower band
        exit_short = (position == -1 and 
                     (low[i] <= bb_lower_20_1d_aligned[i] or  # Take profit at lower band
                      high[i] >= bb_upper_20_1d_aligned[i]))  # Stop loss at upper band
        
        # Priority: breakout > mean reversion > hold
        if breakout_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif mr_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif mr_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals