# 1d_Bollinger_Bands_2_20_Squeeze_Breakout_With_Volume_And_ADX_Filter
# Hypothesis: Bollinger Band squeeze (low volatility breakout) combined with volume surge and ADX > 20 trend filter.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue).
# Uses daily timeframe with weekly EMA filter for higher timeframe bias.
# Target: 30-100 trades over 4 years to minimize fee drag.
# Entry: Price breaks above upper BB with volume spike and ADX > 20, or breaks below lower BB with volume spike and ADX > 20.
# Exit: Price returns to middle Bollinger Band (20-period SMA).
# Position size: 0.25 (25% of capital) to balance return and drawdown.

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
    
    # === Daily Bollinger Bands (20, 2) ===
    df_1d = get_htf_data(prices, '1d')
    # Calculate 20-period SMA and standard deviation
    sma_20 = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    middle_bb = sma_20  # 20-period SMA
    
    # Align Bollinger Bands to 1d timeframe (already 1d, but align for consistency)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb.values)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb.values)
    middle_bb_aligned = align_htf_to_ltf(prices, df_1d, middle_bb.values)
    
    # === Volume Confirmation (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (2.0 * vol_ma)  # Require 2x average volume for breakout
    
    # === Weekly EMA Filter (for higher timeframe bias) ===
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean()
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w.values)
    
    # === Daily ADX for Trend Strength Filter ===
    # Calculate ADX on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed ATR, +DM, -DM
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / np.where(atr == 0, np.nan, atr)
    minus_di = 100 * minus_dm_smooth / np.where(atr == 0, np.nan, atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, np.nan, (plus_di + minus_di))
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 50  # Need BB(20), EMA50, and ADX warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(middle_bb_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        ema50_1w = ema_50_1w_aligned[i]
        adx_val = adx_aligned[i]
        upper = upper_bb_aligned[i]
        lower = lower_bb_aligned[i]
        middle = middle_bb_aligned[i]
        
        # === EXIT LOGIC: Close position when price returns to middle Bollinger Band ===
        if position == 1:  # Long position
            # Exit when price crosses back below middle BB
            if price < middle:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price crosses back above middle BB
            if price > middle:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Trend filter: only trade when ADX > 20 (trending market)
            if adx_val > 20:
                # LONG: Price breaks above upper BB with volume confirmation and price > weekly EMA50 (bullish bias)
                if price > upper and vol_spike and price > ema50_1w:
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Price breaks below lower BB with volume confirmation and price < weekly EMA50 (bearish bias)
                elif price < lower and vol_spike and price < ema50_1w:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Bollinger_Bands_2_20_Squeeze_Breakout_With_Volume_And_ADX_Filter"
timeframe = "1d"
leverage = 1.0