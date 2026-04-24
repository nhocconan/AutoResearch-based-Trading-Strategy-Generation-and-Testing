#!/usr/bin/env python3
"""
Hypothesis: 4h CRSI(2,14,100) with 1d ADX(14) regime filter and volume confirmation.
- Primary timeframe: 4h for entries/exits.
- HTF: 1d ADX(14) for trend strength (ADX > 25 = trending regime).
- Volume: Current 4h volume > 1.5 * 20-period 4h volume MA to confirm momentum.
- Entry: Long when CRSI < 15 AND ADX > 25 AND volume spike.
         Short when CRSI > 85 AND ADX > 25 AND volume spike.
- Exit: Opposite CRSI extreme or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate CRSI (Connors RSI) on 4h: RSI(3) + RSI(UpDown Length,2) + PercentRank(close,100) / 3
    # RSI(3)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/3, adjust=False, min_periods=3).mean()
    avg_loss = loss.ewm(alpha=1/3, adjust=False, min_periods=3).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_3 = 100 - (100 / (1 + rs))
    rsi_3 = rsi_3.fillna(50).values  # neutral when no data
    
    # Up/Down streak for RSI(2)
    up_down = np.where(delta > 0, 1, np.where(delta < 0, -1, 0))
    streak = np.zeros_like(up_down, dtype=int)
    for i in range(1, n):
        if up_down[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif up_down[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    streak_abs = np.abs(streak)
    # RSI(2) on streak
    streak_delta = pd.Series(streak).diff()
    streak_gain = streak_delta.clip(lower=0)
    streak_loss = -streak_delta.clip(upper=0)
    streak_avg_gain = streak_gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    streak_avg_loss = streak_loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    streak_rs = streak_avg_gain / streak_avg_loss.replace(0, np.nan)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.fillna(50).values
    
    # Percent Rank of close over 100 periods
    close_series = pd.Series(close)
    percent_rank = close_series.rolling(window=100, min_periods=1).apply(
        lambda x: np.percentile(x, x.iloc[-1]) if len(x) > 0 else 50, raw=False
    ).values
    percent_rank = np.where(np.isnan(percent_rank), 50, percent_rank)
    
    # CRSI = (RSI(3) + RSI(Streak) + PercentRank) / 3
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    
    # Get 1d data for ADX(14) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d
    df_1d_high = pd.Series(df_1d['high'].values)
    df_1d_low = pd.Series(df_1d['low'].values)
    df_1d_close = pd.Series(df_1d['close'].values)
    
    # True Range
    tr1 = df_1d_high - df_1d_low
    tr2 = np.abs(df_1d_high - df_1d_close.shift(1))
    tr3 = np.abs(df_1d_low - df_1d_close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Movement
    up_move = df_1d_high.diff()
    down_move = df_1d_low.diff().multiply(-1)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_1d)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_1d)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_1d = adx_1d.fillna(0).values
    
    # Align HTF ADX to 4h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: current 4h volume > 1.5 * 20-period 4h volume MA
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 30)  # Need enough bars for CRSI percent rank and 1d ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(crsi[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        crsi_val = crsi[i]
        adx_val = adx_1d_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with volume spike and ADX filter
            if volume_spike[i] and adx_val > 25:
                # Oversold: CRSI < 15 -> long
                if crsi_val < 15:
                    signals[i] = 0.25
                    position = 1
                # Overbought: CRSI > 85 -> short
                elif crsi_val > 85:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: CRSI > 70 (overbought) OR loss of volume confirmation OR ADX weakens
            if crsi_val > 70 or not volume_spike[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: CRSI < 30 (oversold) OR loss of volume confirmation OR ADX weakens
            if crsi_val < 30 or not volume_spike[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_CRSI2_14_100_ADX14_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0