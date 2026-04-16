#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volume filter (volume > 1.5x ATR-scaled median) and 1w EMA trend filter (price > EMA50)
# Long when price > Donchian upper band AND 1d volume > 1.5x (ATR(14) * close median) AND weekly close > weekly EMA50
# Short when price < Donchian lower band AND 1d volume > 1.5x (ATR(14) * close median) AND weekly close < weekly EMA50
# Exit when price crosses Donchian middle band
# Uses discrete position size 0.25. Target: 75-200 total trades over 4 years.
# Combines price channel breakout with volatility-adjusted volume confirmation and weekly trend filter for robustness in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Donchian levels and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Donchian channels (20-period) and ATR-scaled volume median ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Donchian channels
    donchian_upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_middle_20 = (donchian_upper_20 + donchian_lower_20) / 2.0
    
    # ATR(14) for 1d
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Close median for scaling
    close_median_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).median().values
    vol_threshold_base = atr_1d * close_median_20_1d  # ATR-scaled base threshold
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Weekly Indicators: EMA50 trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_20)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle_20)
    vol_threshold_base_aligned = align_htf_to_ltf(prices, df_1d, vol_threshold_base)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, 50)  # 1d Donchian, 1d ATR/EMA, weekly EMA50
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(vol_threshold_base_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values (aligned)
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        middle = donchian_middle_aligned[i]
        vol_threshold_base_val = vol_threshold_base_aligned[i]
        weekly_ema50 = ema_50_1w_aligned[i]
        
        # Current 1d volume (aligned)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        if np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1d volume > 1.5x ATR-scaled base threshold
        vol_threshold = vol_threshold_base_val * 1.5
        vol_confirm = vol_1d_aligned[i] > vol_threshold
        
        # Weekly trend filter: price above/below EMA50 indicates trend direction
        weekly_trend_up = close_1d[-1] > weekly_ema50 if len(close_1d) > 0 else False  # placeholder, will use aligned weekly close below
        # Get aligned weekly close for proper comparison
        df_1w_close = df_1w['close'].values
        weekly_close_aligned = align_htf_to_ltf(prices, df_1w, df_1w_close)
        weekly_trend_up = weekly_close_aligned[i] > weekly_ema50
        weekly_trend_down = weekly_close_aligned[i] < weekly_ema50
        
        # Price levels
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price crosses below Donchian middle band (mean reversion)
            if price < middle:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price crosses above Donchian middle band (mean reversion)
            if price > middle:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Price breaks above Donchian upper band AND volume confirmation AND weekly uptrend
            if price > upper and vol_confirm and weekly_trend_up:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below Donchian lower band AND volume confirmation AND weekly downtrend
            elif price < lower and vol_confirm and weekly_trend_down:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_1dATR_VolumeSpike1.5x_1wEMA50_v1"
timeframe = "4h"
leverage = 1.0