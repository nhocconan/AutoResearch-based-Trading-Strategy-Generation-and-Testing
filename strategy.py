# 1d_Camarilla_R1_S1_Breakout_1wEMA34_Volume_Filter
# Hypothesis: 1d Camarilla pivot breakouts with weekly EMA trend filter and volume confirmation
# Work in bull markets via breakout momentum and in bear via mean reversion at pivot levels
# Target: 10-25 trades/year to avoid fee drag, focus on high-probability setups

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
    
    # Get 1d data once for primary analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1d OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels (using prior day's range)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R1, S1 based on previous day
    range_prev = prev_high - prev_low
    R1 = prev_close + (range_prev * 1.0 / 12)
    S1 = prev_close - (range_prev * 1.0 / 12)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Weekly EMA34 for higher timeframe trend
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to lower timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Align weekly EMA to lower timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation (1d volume > 20-day average)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = df_1d['volume'].values / vol_ma_1d
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Price relative to Camarilla levels
        price_above_R1 = close[i] > R1_aligned[i]
        price_below_S1 = close[i] < S1_aligned[i]
        
        # Trend filters: daily and weekly alignment
        daily_uptrend = close[i] > ema_34_1d_aligned[i]
        weekly_uptrend = close[i] > ema_34_1w_aligned[i]
        daily_downtrend = close[i] < ema_34_1d_aligned[i]
        weekly_downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Volume confirmation: require significantly above-average volume
        vol_confirm = vol_ratio_aligned[i] > 1.5
        
        # Entry conditions - selective for quality
        # Long: price breaks above R1 with daily/weekly uptrend and volume
        long_entry = price_above_R1 and daily_uptrend and weekly_uptrend and vol_confirm
        # Short: price breaks below S1 with daily/weekly downtrend and volume
        short_entry = price_below_S1 and daily_downtrend and weekly_downtrend and vol_confirm
        
        # Exit conditions: return to mean (opposite pivot level) or trend breakdown
        long_exit = close[i] < S1_aligned[i] or not (daily_uptrend and weekly_uptrend)
        short_exit = close[i] > R1_aligned[i] or not (daily_downtrend and weekly_downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wEMA34_Volume_Filter"
timeframe = "1d"
leverage = 1.0