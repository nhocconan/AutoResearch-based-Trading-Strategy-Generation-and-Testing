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
    
    # Load weekly data for ADX and RSI - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 14:
        return np.zeros(n)
    
    # Calculate weekly RSI(14)
    weekly_close = df_weekly['close'].values
    delta = np.diff(weekly_close, prepend=weekly_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    weekly_rsi = 100 - (100 / (1 + rs))
    
    # Calculate weekly ADX(14)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close_adx = df_weekly['close'].values
    
    # True Range
    tr1 = weekly_high - weekly_low
    tr2 = np.abs(weekly_high - np.roll(weekly_close_adx, 1))
    tr3 = np.abs(weekly_low - np.roll(weekly_close_adx, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((weekly_high - np.roll(weekly_high, 1)) > (np.roll(weekly_low, 1) - weekly_low),
                       np.maximum(weekly_high - np.roll(weekly_high, 1), 0), 0)
    dm_minus = np.where((np.roll(weekly_low, 1) - weekly_low) > (weekly_high - np.roll(weekly_high, 1)),
                        np.maximum(np.roll(weekly_low, 1) - weekly_low, 0), 0)
    
    # Smooth TR, DM+
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    weekly_adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly indicators to daily
    weekly_rsi_aligned = align_htf_to_ltf(prices, df_weekly, weekly_rsi)
    weekly_adx_aligned = align_htf_to_ltf(prices, df_weekly, weekly_adx)
    
    # Daily RSI(14) for entry timing
    delta_daily = np.diff(close, prepend=close[0])
    gain_daily = np.where(delta_daily > 0, delta_daily, 0)
    loss_daily = np.where(delta_daily < 0, -delta_daily, 0)
    avg_gain_daily = pd.Series(gain_daily).rolling(window=14, min_periods=14).mean().values
    avg_loss_daily = pd.Series(loss_daily).rolling(window=14, min_periods=14).mean().values
    rs_daily = avg_gain_daily / (avg_loss_daily + 1e-10)
    daily_rsi = 100 - (100 / (1 + rs_daily))
    
    # Daily volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(weekly_rsi_aligned[i]) or np.isnan(weekly_adx_aligned[i]) or 
            np.isnan(daily_rsi[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly regime filter: ADX > 25 for trending, RSI not extreme
        adx = weekly_adx_aligned[i]
        weekly_rsi_val = weekly_rsi_aligned[i]
        trending = adx > 25
        not_overbought = weekly_rsi_val < 70
        not_oversold = weekly_rsi_val > 30
        
        if position == 0:
            # Long: Daily RSI oversold (<30) in weekly uptrend with volume
            if (daily_rsi[i] < 30 and weekly_rsi_val > 50 and trending and
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Daily RSI overbought (>70) in weekly downtrend with volume
            elif (daily_rsi[i] > 70 and weekly_rsi_val < 50 and trending and
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Daily RSI returns to neutral zone (40-60)
            if position == 1:
                if daily_rsi[i] > 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if daily_rsi[i] < 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_WeeklyTrend_DailyRSI_Volume"
timeframe = "1d"
leverage = 1.0