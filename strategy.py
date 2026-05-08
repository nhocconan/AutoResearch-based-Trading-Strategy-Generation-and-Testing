#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h CRSI with Daily Trend and Volume Confirmation
# Uses 12h CRSI (Composite RSI) for mean reversion entries, filtered by daily EMA34 trend and volume spike (>2x 20-period average).
# Designed to capture reversals in both bull and bear markets with high win rate.
# Target: 12-37 trades/year (50-150 total over 4 years). Uses CRSI for precise entries and trend/volume filters to reduce whipsaw.

name = "12h_CRSI_DailyTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA34 trend
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend
    close_daily = df_daily['close'].values
    ema34_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 34:
        ema34_daily[33] = np.mean(close_daily[:34])
        for i in range(34, len(close_daily)):
            ema34_daily[i] = (close_daily[i] * 2 + ema34_daily[i-1] * 32) / 34
    
    # Calculate 12h RSI(3) for CRSI
    def rsi(close_arr, period):
        rsi_arr = np.full(len(close_arr), np.nan)
        if len(close_arr) < period:
            return rsi_arr
        delta = np.diff(close_arr)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full(len(close_arr), np.nan)
        avg_loss = np.full(len(close_arr), np.nan)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period + 1, len(close_arr)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = avg_gain / avg_loss
        rsi_arr = 100 - (100 / (1 + rs))
        return rsi_arr
    
    rsi3 = rsi(close, 3)
    
    # Calculate 12h RSI(2) for streak
    rsi2 = rsi(close, 2)
    
    # Calculate streak RSI (consecutive up/down days)
    streak_rsi = np.full(n, np.nan)
    up_days = np.zeros(n)
    down_days = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            up_days[i] = up_days[i-1] + 1
            down_days[i] = 0
        elif close[i] < close[i-1]:
            down_days[i] = down_days[i-1] + 1
            up_days[i] = 0
        else:
            up_days[i] = up_days[i-1]
            down_days[i] = down_days[i-1]
        streak_rsi[i] = rsi2[i] if up_days[i] >= 2 or down_days[i] >= 2 else 50  # neutral if no streak
    
    # Calculate Percent Rank of RSI(3) over 100 periods
    def percent_rank(arr, window):
        pr = np.full(len(arr), np.nan)
        for i in range(window, len(arr)):
            window_data = arr[i-window:i]
            if np.all(np.isnan(window_data)):
                pr[i] = np.nan
            else:
                valid_data = window_data[~np.isnan(window_data)]
                if len(valid_data) == 0:
                    pr[i] = np.nan
                else:
                    score = arr[i]
                    if np.isnan(score):
                        pr[i] = np.nan
                    else:
                        pr[i] = np.sum(valid_data < score) / len(valid_data) * 100
        return pr
    
    pr_rsi3 = percent_rank(rsi3, 100)
    
    # Calculate CRSI: (RSI(3) + RSI(Streak) + PercentRank(RSI(3))) / 3
    crsi = (rsi3 + streak_rsi + pr_rsi3) / 3
    
    # Calculate 12h volume average for volume spike
    vol_avg_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Align daily EMA34 to 12h timeframe
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 100)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(crsi[i]) or np.isnan(ema34_daily_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Find current daily bar's close
        close_daily_current = np.nan
        if not np.isnan(ema34_daily_aligned[i]):
            idx_daily = 0
            while idx_daily < len(df_daily) and df_daily.iloc[idx_daily]['open_time'] <= prices.iloc[i]['open_time']:
                idx_daily += 1
            idx_daily -= 1  # last completed daily bar
            
            if idx_daily >= 0:
                close_daily_current = df_daily.iloc[idx_daily]['close']
        
        if np.isnan(close_daily_current):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check conditions
        price_above_ema = close_daily_current > ema34_daily_aligned[i]
        price_below_ema = close_daily_current < ema34_daily_aligned[i]
        vol_spike = volume[i] > 2.0 * vol_avg_20[i]
        
        if position == 0:
            # Look for entry: CRSI oversold/overbought with trend and volume confirmation
            if crsi[i] < 15 and price_above_ema and vol_spike:
                signals[i] = 0.25
                position = 1
            elif crsi[i] > 85 and price_below_ema and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: CRSI overbought or trend fails or volume drops
            if crsi[i] > 80 or not price_above_ema or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: CRSI oversold or trend fails or volume drops
            if crsi[i] < 20 or not price_below_ema or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals