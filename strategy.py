#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volume confirmation and 1w EMA50 trend filter.
# Long when price > upper band, 1d ATR(14) > 1.5x its 20-period median (volatility expansion), and weekly close > weekly EMA50.
# Short when price < lower band, same volatility expansion condition, and weekly close < weekly EMA50.
# Exit when price crosses middle band (mean reversion).
# Uses discrete position size 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Combines price channel breakout with volatility expansion filter and weekly trend filter for robustness.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Donchian levels and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Donchian channels (20-period) and ATR(14) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channels
    donchian_upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_middle_20 = (donchian_upper_20 + donchian_lower_20) / 2.0
    
    # ATR(14) calculation
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR median for volatility expansion filter
    atr_median_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).median().values
    
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
    atr_median_aligned = align_htf_to_ltf(prices, df_1d, atr_median_20)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Align 1d ATR for volatility confirmation
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, 50)  # 1d Donchian, 1d ATR median, weekly EMA50
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(atr_median_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values (aligned)
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        middle = donchian_middle_aligned[i]
        atr_median = atr_median_aligned[i]
        weekly_ema50 = ema_50_1w_aligned[i]
        atr_14 = atr_14_aligned[i]
        
        # Get aligned weekly close for proper trend comparison
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
            # Volatility expansion filter: current 1d ATR(14) > 1.5x its 20-period median
            vol_expansion = atr_14 > (atr_median * 1.5)
            
            # LONG CONDITIONS
            # Price breaks above Donchian upper band AND volatility expansion AND weekly uptrend
            if price > upper and vol_expansion and weekly_trend_up:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below Donchian lower band AND volatility expansion AND weekly downtrend
            elif price < lower and vol_expansion and weekly_trend_down:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_1dATRVoltExp1.5x_1wEMA50_v1"
timeframe = "4h"
leverage = 1.0