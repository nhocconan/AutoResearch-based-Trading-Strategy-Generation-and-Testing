#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot + Daily Trend + Volume Spike
# Long when price breaks above weekly R1 AND daily close > EMA50 (bullish trend) AND volume > 2.0x 20-bar average.
# Short when price breaks below weekly S1 AND daily close < EMA50 (bearish trend) AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Weekly pivots provide strong structural support/resistance, daily EMA50 filters trend alignment, volume spike confirms momentum.
# Primary timeframe: 6h, HTF: 1w for pivots, 1d for EMA trend.

name = "6h_WeeklyPivot_R1S1_Breakout_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Weekly Pivot levels (using previous week's OHLC)
    # Standard pivot: P = (H+L+C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    prev_weekly_high = df_1w['high'].values
    prev_weekly_low = df_1w['low'].values
    prev_weekly_close = df_1w['close'].values
    
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - prev_weekly_low
    weekly_s1 = 2 * weekly_pivot - prev_weekly_high
    
    # Align Weekly Pivot levels to 6h timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Daily EMA50 trend filter
    ema_50 = pd.Series(prev_weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values  # Using weekly close for stability
    # Actually, let's use daily close for EMA50
    prev_daily_close = df_1d['close'].values
    ema_50 = pd.Series(prev_daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: current 6h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for EMA and indicators
    
    for i in range(start_idx, n):
        if np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)  # Volume spike threshold
        
        # Weekly Pivot breakout signals
        breakout_up = curr_high > weekly_r1_aligned[i]  # break above R1
        breakout_down = curr_low < weekly_s1_aligned[i]  # break below S1
        
        # Trend filter: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = curr_close > ema_50_aligned[i]
        bearish_trend = curr_close < ema_50_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R1 AND bullish trend AND volume confirmation
            if (breakout_up and 
                bullish_trend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 AND bearish trend AND volume confirmation
            elif (breakout_down and 
                  bearish_trend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below S1 (stoploss) OR trend turns bearish
            if (curr_low < weekly_s1_aligned[i] or 
                bearish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above R1 (stoploss) OR trend turns bullish
            if (curr_high > weekly_r1_aligned[i] or 
                bullish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals