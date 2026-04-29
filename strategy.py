#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation + ATR stoploss
# Long when price breaks above Donchian(20) upper band AND price > 1w EMA50 AND volume > 1.5x 20-bar avg
# Short when price breaks below Donchian(20) lower band AND price < 1w EMA50 AND volume > 1.5x 20-bar avg
# Exit when price crosses Donchian(20) midpoint (mean reversion) OR ATR-based stoploss
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 7-25 trades/year on 1d timeframe.
# Donchian channels provide structure, 1w EMA50 filters counter-trend moves, volume confirms strength.
# Works in bull via trend alignment, in bear via short signals. ATR stoploss manages risk.

name = "1d_Donchian20_1wEMA50_VolumeConfirm_v3"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Donchian(20) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Align Donchian levels to 1d timeframe (already aligned, but using helper for consistency)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle)
    
    # Calculate ATR(14) for stoploss
    tr1 = pd.Series(high_1d - low_1d).values
    tr2 = pd.Series(np.abs(high_1d - np.concatenate([[np.nan], close_1d[:-1]]))).values
    tr3 = pd.Series(np.abs(low_1d - np.concatenate([[np.nan], close_1d[:-1]]))).values
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w data
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 20, 50, 14)  # Donchian, volume MA, EMA50, ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_upper = donchian_upper_aligned[i]
        curr_lower = donchian_lower_aligned[i]
        curr_middle = donchian_middle_aligned[i]
        curr_ema50 = ema_50_1w_aligned[i]
        curr_atr = atr_14_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses Donchian midpoint OR ATR stoploss hit
            if curr_close < curr_middle or curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses Donchian midpoint OR ATR stoploss hit
            if curr_close > curr_middle or curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian upper AND price > 1w EMA50 AND volume confirmation
            if curr_close > curr_upper and curr_close > curr_ema50 and vol_conf:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short when price breaks below Donchian lower AND price < 1w EMA50 AND volume confirmation
            elif curr_close < curr_lower and curr_close < curr_ema50 and vol_conf:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals