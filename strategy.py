#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 4h ATR-based volume confirmation and 1d EMA50 trend filter.
# Long when price > upper band, 4h ATR(14) > 1.5x its 20-period median (volatility expansion), and daily close > daily EMA50.
# Short when price < lower band, same volatility expansion condition, and daily close < daily EMA50.
# Exit when price crosses middle band (mean reversion).
# Uses discrete position size 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Combines price channel breakout with volatility expansion filter and daily trend filter for robustness.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop for Donchian levels and ATR calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: Donchian channels (20-period) and ATR(14) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian channels
    donchian_upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_middle_20 = (donchian_upper_20 + donchian_lower_20) / 2.0
    
    # ATR(14) calculation
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR median for volatility expansion filter
    atr_median_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).median().values
    
    # Get daily data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === Daily Indicators: EMA50 trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_20)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle_20)
    atr_median_aligned = align_htf_to_ltf(prices, df_4h, atr_median_20)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Align 4h ATR for volatility confirmation
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, 50)  # 4h Donchian, 4h ATR median, daily EMA50
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(atr_median_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values (aligned)
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        middle = donchian_middle_aligned[i]
        atr_median = atr_median_aligned[i]
        daily_ema50 = ema_50_1d_aligned[i]
        atr_14 = atr_14_aligned[i]
        
        # Get aligned daily close for proper trend comparison
        df_1d_close = df_1d['close'].values
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, df_1d_close)
        daily_trend_up = daily_close_aligned[i] > daily_ema50
        daily_trend_down = daily_close_aligned[i] < daily_ema50
        
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
            # Volatility expansion filter: current 4h ATR(14) > 1.5x its 20-period median
            vol_expansion = atr_14 > (atr_median * 1.5)
            
            # LONG CONDITIONS
            # Price breaks above Donchian upper band AND volatility expansion AND daily uptrend
            if price > upper and vol_expansion and daily_trend_up:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below Donchian lower band AND volatility expansion AND daily downtrend
            elif price < lower and vol_expansion and daily_trend_down:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_4hATRVoltExp1.5x_1dEMA50_v1"
timeframe = "4h"
leverage = 1.0