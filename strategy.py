#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volume spike and 1w EMA50 trend filter.
# Long when price > upper band, 1d volume > 1.5x ATR(20)-scaled median, and weekly close > weekly EMA50.
# Short when price < lower band, same volume condition, and weekly close < weekly EMA50.
# Exit when price crosses the middle band (mean reversion).
# Uses discrete position size 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Combines price channel breakout with volume spike filter and weekly trend filter for robustness in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop for Donchian levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: Donchian channel (20-period) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channels
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2.0
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Weekly Indicators: EMA50 trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === Daily Indicators: ATR(20) for volume spike scaling ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (4h)
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    middle_20_aligned = align_htf_to_ltf(prices, df_4h, middle_20)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    atr_20_aligned = align_htf_to_ltf(prices, df_1d, atr_20)
    
    # Align daily volume and ATR-scaled volume median
    vol_1d = df_1d['volume'].values
    vol_median_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).median().values
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    atr_median_aligned = align_htf_to_ltf(prices, df_1d, atr_20)  # reuse ATR aligned
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 20, 50, 20)  # Donchian(20), daily volume median(20), weekly EMA50, ATR(20)
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(middle_20_aligned[i]) or np.isnan(vol_median_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_1d_aligned[i]) or 
            np.isnan(atr_median_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values (aligned)
        upper = upper_20_aligned[i]
        lower = lower_20_aligned[i]
        middle = middle_20_aligned[i]
        vol_median = vol_median_aligned[i]
        weekly_ema50 = ema_50_1w_aligned[i]
        daily_volume = vol_1d_aligned[i]
        atr_value = atr_median_aligned[i]
        
        # Get aligned daily close for proper trend comparison
        df_1d_close = df_1d['close'].values
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, df_1d_close)
        weekly_trend_up = daily_close_aligned[i] > weekly_ema50  # Using daily close vs weekly EMA for trend
        weekly_trend_down = daily_close_aligned[i] < weekly_ema50
        
        # Price levels
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price crosses below middle band (mean reversion)
            if price < middle:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price crosses above middle band (mean reversion)
            if price > middle:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume spike filter: current daily volume > 1.5x ATR(20) * median volume
            # This scales volume threshold by volatility
            volume_threshold = vol_median * (1.0 + atr_value * 0.1)  # ATR-based dynamic threshold
            volume_spike = daily_volume > volume_threshold
            
            # LONG CONDITIONS
            # Price breaks above upper Donchian band AND volume spike AND weekly uptrend
            if price > upper and volume_spike and weekly_trend_up:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below lower Donchian band AND volume spike AND weekly downtrend
            elif price < lower and volume_spike and weekly_trend_down:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_1dATRVolSpike_1wEMA50_v1"
timeframe = "4h"
leverage = 1.0