# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe with 1h market regime filter and weekly ADX filter.
Trade mean reversion at 1d RSI extremes only when 1h momentum is weak (ADX<25) and 1h price is inside Bollinger Bands.
Use weekly ADX>25 to ensure we only trade when higher timeframe trend is strong enough to sustain mean reversion moves.
Position size: 0.25. Target: 15-25 trades/year.
Works in bull markets via buying oversold dips in uptrend and in bear via selling overbought rallies in downtrend.
"""
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
    
    # Get 1h data for regime filter (ADX and Bollinger Bands)
    df_1h = get_htf_data(prices, '1h')
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate 1h ADX(14) for trend strength
    # TR = max(high-low, |high-previous close|, |low-previous close|)
    tr1 = high_1h - low_1h
    tr2 = np.abs(high_1h - np.roll(close_1h, 1))
    tr3 = np.abs(low_1h - np.roll(close_1h, 1))
    tr1[0] = high_1h[0] - low_1h[0]  # first period
    tr2[0] = np.abs(high_1h[0] - close_1h[0])
    tr3[0] = np.abs(low_1h[0] - close_1h[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM
    up_move = np.diff(high_1h, prepend=high_1h[0])
    down_move = -np.diff(low_1h, prepend=low_1h[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, TR
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # 1h Bollinger Bands (20, 2)
    bb_middle = pd.Series(close_1h).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1h).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Get 1d data for entry signal (RSI)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Get weekly data for trend filter (ADX)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly ADX(14) for trend strength filter
    tr1w = high_1w - low_1w
    tr2w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3w = np.abs(low_1w - np.roll(close_1w, 1))
    tr1w[0] = high_1w[0] - low_1w[0]
    tr2w[0] = np.abs(high_1w[0] - close_1w[0])
    tr3w[0] = np.abs(low_1w[0] - close_1w[0])
    trw = np.maximum(tr1w, np.maximum(tr2w, tr3w))
    atrw = pd.Series(trw).rolling(window=14, min_periods=14).mean().values
    
    up_move_w = np.diff(high_1w, prepend=high_1w[0])
    down_move_w = -np.diff(low_1w, prepend=low_1w[0])
    plus_dm_w = np.where((up_move_w > down_move_w) & (up_move_w > 0), up_move_w, 0.0)
    minus_dm_w = np.where((down_move_w > up_move_w) & (down_move_w > 0), down_move_w, 0.0)
    
    plus_di_w = 100 * pd.Series(plus_dm_w).rolling(window=14, min_periods=14).mean().values / atrw
    minus_di_w = 100 * pd.Series(minus_dm_w).rolling(window=14, min_periods=14).mean().values / atrw
    dx_w = 100 * np.abs(plus_di_w - minus_di_w) / (plus_di_w + minus_di_w + 1e-10)
    adx_1w = pd.Series(dx_w).rolling(window=14, min_periods=14).mean().values
    
    # Align 1h and weekly data to 1d
    adx_1h_aligned = align_htf_to_ltf(prices, df_1h, adx)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1h, bb_lower)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_1h_aligned[i]) or np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or np.isnan(adx_1w_aligned[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) AND 1h momentum weak (ADX<25) AND price above BB lower (not extreme oversold)
            # AND weekly trend strong (ADX>25) to ensure mean reversion has room to run
            if (rsi[i] < 30 and adx_1h_aligned[i] < 25 and close[i] > bb_lower_aligned[i] and 
                adx_1w_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) AND 1h momentum weak (ADX<25) AND price below BB upper (not extreme overbought)
            # AND weekly trend strong (ADX>25)
            elif (rsi[i] > 70 and adx_1h_aligned[i] < 25 and close[i] < bb_upper_aligned[i] and 
                  adx_1w_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (>50) or weekly trend weakens (ADX<20)
            if rsi[i] > 50 or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral (<50) or weekly trend weakens (ADX<20)
            if rsi[i] < 50 or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1hADX_BB_RSI_1wADX"
timeframe = "1d"
leverage = 1.0