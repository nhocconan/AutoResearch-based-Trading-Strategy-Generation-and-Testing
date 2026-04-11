#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily ADX trend strength combined with weekly Bollinger Band squeeze.
# Enters long when weekly Bollinger Bands are tight (low volatility) and daily ADX > 25 (trending up).
# Enters short when weekly Bollinger Bands are tight and daily ADX > 25 with negative DI crossover.
# Uses Bollinger Band width percentile to detect squeeze and ADX for trend confirmation.
# Designed for low-frequency, high-conviction trades (target: 15-40 trades/year).
# Weekly volatility filter prevents entries during high-chop periods; ADX ensures directional strength.

name = "1d_1w_bbwidth_adx_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly Bollinger Bands (20, 2)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Typical price for BB
    tp_1w = (high_1w + low_1w + close_1w) / 3
    sma_20 = np.full_like(tp_1w, np.nan)
    std_20 = np.full_like(tp_1w, np.nan)
    
    for i in range(19, len(tp_1w)):
        sma_20[i] = np.mean(tp_1w[i-19:i+1])
        std_20[i] = np.std(tp_1w[i-19:i+1])
    
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # BB width percentile (252-week lookback ~ 5 years)
    bb_width_pct = np.full_like(bb_width, np.nan)
    lookback = 252
    for i in range(lookback, len(bb_width)):
        window = bb_width[i-lookback:i+1]
        if not np.all(np.isnan(window)):
            bb_width_pct[i] = np.percentile(window[~np.isnan(window)], 10)  # 10th percentile
    
    # Squeeze condition: BB width <= 10th percentile (low volatility)
    squeeze = bb_width <= bb_width_pct
    
    # Align squeeze to daily
    squeeze_aligned = align_htf_to_ltf(prices, df_1w, squeeze)
    
    # Daily ADX (14-period)
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
    
    # Smoothed TR and DM
    atr = np.full_like(tr, np.nan)
    plus_dm_sm = np.full_like(tr, np.nan)
    minus_dm_sm = np.full_like(tr, np.nan)
    
    # Wilder's smoothing (alpha = 1/14)
    alpha = 1/14
    for i in range(1, len(tr)):
        if np.isnan(tr[i-1]):
            atr[i] = tr[i]
            plus_dm_sm[i] = plus_dm[i]
            minus_dm_sm[i] = minus_dm[i]
        else:
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
            plus_dm_sm[i] = alpha * plus_dm[i] + (1 - alpha) * plus_dm_sm[i-1]
            minus_dm_sm[i] = alpha * minus_dm[i] + (1 - alpha) * minus_dm_sm[i-1]
    
    # DI values
    plus_di = 100 * plus_dm_sm / atr
    minus_di = 100 * minus_dm_sm / atr
    
    # DX and ADX
    dx = np.full_like(tr, np.nan)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    
    adx = np.full_like(tr, np.nan)
    for i in range(14, len(dx)):
        if np.isnan(adx[i-1]):
            adx[i] = np.mean(dx[i-13:i+1])
        else:
            adx[i] = alpha * dx[i] + (1 - alpha) * adx[i-1]
    
    # Align ADX and DI to daily
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    # Volume filter: 20-day average volume
    volume_1d = df_1d['volume'].values
    vol_avg_20 = np.full_like(volume_1d, np.nan)
    for i in range(19, len(volume_1d)):
        vol_avg_20[i] = np.mean(volume_1d[i-19:i+1])
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(squeeze_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Conditions
        is_squeeze = squeeze_aligned[i]
        strong_trend = adx_aligned[i] > 25
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        di_cross_up = plus_di_aligned[i] > minus_di_aligned[i]
        di_cross_down = minus_di_aligned[i] > plus_di_aligned[i]
        
        # Entry conditions
        long_entry = is_squeeze and strong_trend and vol_filter and di_cross_up
        short_entry = is_squeeze and strong_trend and vol_filter and di_cross_down
        
        # Exit conditions: loss of squeeze or trend weakness
        exit_long = position == 1 and (not is_squeeze or adx_aligned[i] < 20)
        exit_short = position == -1 and (not is_squeeze or adx_aligned[i] < 20)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals