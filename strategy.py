#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dRegime_VolumeSpike
Hypothesis: Uses 1h Camarilla pivot breakouts filtered by 4h trend and 1d choppy regime.
Enter long when 1h price breaks above R1, 4h close > 4h EMA20 (uptrend), 1d chop > 50 (range market), and volume > 1.5x average.
Enter short when 1h price breaks below S1, 4h close < 4h EMA20 (downtrend), 1d chop > 50, and volume > 1.5x average.
Exit when price returns to 1h pivot (PP) or 4h trend reverses.
Designed for 1h timeframe with tight entries to avoid fee drag: target 15-35 trades/year.
Works in both bull and bear markets via 4h trend filter and 1d regime filter (choppy markets favor mean reversion at pivots).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1h data for Camarilla pivots and entry timing
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate Camarilla pivots on 1h data (using previous 1h bar's OHLC)
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Shift by 1 to get previous 1h bar's OHLC for current Camarilla levels
    prev_high = np.roll(high_1h, 1)
    prev_low = np.roll(low_1h, 1)
    prev_close = np.roll(close_1h, 1)
    # First value will be invalid (rolled from last), set to nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_pp = (prev_high + prev_low + prev_close) / 3.0
    camarilla_range = prev_high - prev_low
    camarilla_r1 = camarilla_pp + (camarilla_range * 1.1 / 4.0)
    camarilla_s1 = camarilla_pp - (camarilla_range * 1.1 / 4.0)
    
    # Align 1h Camarilla levels to 1h timeframe (identity alignment)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1h, camarilla_s1)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1h, camarilla_pp)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA20 for trend filter
    close_4h_series = pd.Series(df_4h['close'].values)
    ema_20_4h = close_4h_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1d data for choppy regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Choppiness Index (CHOP) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # First value has no previous close
    
    # ATR(14) - sum of TR over 14 periods
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(atr_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero and log of zero
    hh_ll = hh_14 - ll_14
    chop_raw = np.where((hh_ll > 0) & ~np.isnan(hh_ll), 
                        atr_14 / hh_ll, 
                        np.nan)
    chop = 100 * np.log10(chop_raw) / np.log10(14)
    
    # Align 1d Chop to 1h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need 1h data shifted (1), 4h EMA20 (20), 1d CHOP (14), volume avg (20)
    start_idx = max(1, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_4h_val = ema_20_4h_aligned[i]
        chop_val = chop_aligned[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        pp_level = camarilla_pp_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Look for entry: breakout of Camarilla R1/S1 levels with 4h trend filter, 1d chop > 50 (range), and volume
            # Long: price breaks above R1 AND 4h uptrend AND chop > 50 AND volume AND session
            long_condition = (close_val > r1_level) and (close_val > ema_4h_val) and (chop_val > 50) and vol_conf and in_session
            # Short: price breaks below S1 AND 4h downtrend AND chop > 50 AND volume AND session
            short_condition = (close_val < s1_level) and (close_val < ema_4h_val) and (chop_val > 50) and vol_conf and in_session
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to pivot level OR 4h trend breaks
            exit_condition = (close_val <= pp_level) or (close_val < ema_4h_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to pivot level OR 4h trend breaks
            exit_condition = (close_val >= pp_level) or (close_val > ema_4h_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dRegime_VolumeSpike"
timeframe = "1h"
leverage = 1.0