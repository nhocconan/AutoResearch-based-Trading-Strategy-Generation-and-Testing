#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_RSI_Crsi_Reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI and CRSI components
    df_1d = get_htf_data(prices, '1d')
    
    # Get 1w data for Choppiness filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate RSI(3) for CRSI
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=3, min_periods=3).mean()
    avg_loss = loss.rolling(window=3, min_periods=3).mean()
    rs = avg_gain / avg_loss
    rsi_3 = 100 - (100 / (1 + rs))
    rsi_3_values = rsi_3.values
    
    # Calculate Streak RSI (consecutive up/down days)
    change = np.diff(close_1d, prepend=close_1d[0])
    up_days = np.where(change >= 0, 1, 0)
    down_days = np.where(change < 0, 1, 0)
    
    # Streak calculation
    streak_up = np.zeros_like(close_1d)
    streak_down = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        if change[i] >= 0:
            streak_up[i] = streak_up[i-1] + 1
            streak_down[i] = 0
        else:
            streak_down[i] = streak_down[i-1] + 1
            streak_up[i] = 0
    
    # RSI of streak
    streak_up_series = pd.Series(streak_up)
    streak_down_series = pd.Series(streak_down)
    
    avg_gain_up = streak_up_series.rolling(window=2, min_periods=2).mean()
    avg_loss_up = streak_down_series.rolling(window=2, min_periods=2).mean()
    rs_up = avg_gain_up / avg_loss_up
    rsi_streak_up = 100 - (100 / (1 + rs_up))
    
    avg_gain_down = streak_down_series.rolling(window=2, min_periods=2).mean()
    avg_loss_down = streak_up_series.rolling(window=2, min_periods=2).mean()
    rs_down = avg_gain_down / avg_loss_down
    rsi_streak_down = 100 - (100 / (1 + rs_down))
    
    # Combine for CRSI: RSI(3) + RSI_streak(2) + PercentRank(100)
    # Percent rank of RSI(3) over 100 periods
    rsi_3_series = pd.Series(rsi_3_values)
    percent_rank = rsi_3_series.rolling(window=100, min_periods=1).apply(
        lambda x: np.percentile(x, len(x)-1) if len(x) > 0 else 0, raw=False
    )
    # Simpler: percentage of values below current
    percent_rank = rsi_3_series.rolling(window=100, min_periods=1).apply(
        lambda x: (x < x.iloc[-1]).sum() / len(x) * 100 if len(x) > 0 else 50, raw=False
    )
    percent_rank_values = percent_rank.fillna(50).values
    
    crsi = (rsi_3_values + rsi_streak_up + rsi_streak_down + percent_rank_values) / 4
    
    # Calculate Choppiness Index on 1w
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = np.nancumsum(atr_1w) - np.nancumsum(np.where(np.arange(len(atr_1w)) < 14, 0, atr_1w))
    sum_atr_14 = np.where(np.arange(len(atr_1w)) < 13, np.nan, sum_atr_14)
    
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(sum_atr_14 / (highest_high - lowest_low)) / np.log10(14)
    
    # Align indicators to 1d timeframe
    crsi_aligned = align_htf_to_ltf(prices, df_1d, crsi)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(crsi_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Chop filter: trade in ranging markets (Chop > 61.8) or strong trends (Chop < 38.2)
        # For mean reversion, we prefer ranging markets
        chop_filter = chop_aligned[i] > 61.8  # Strong ranging market
        
        if position == 0:
            # Long when CRSI is oversold (<15) in ranging market
            if (crsi_aligned[i] < 15 and chop_filter):
                signals[i] = 0.25
                position = 1
            # Short when CRSI is overbought (>85) in ranging market
            elif (crsi_aligned[i] > 85 and chop_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when CRSI crosses above 50 (mean reversion complete)
            if crsi_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when CRSI crosses below 50
            if crsi_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals