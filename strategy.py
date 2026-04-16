# Hypothesis: The 4h timeframe benefits from trend-following with trend confirmation and volatility filters to avoid whipsaws. Using ADX for trend strength and Bollinger Bands for volatility regime (squeeze/expansion) improves entry quality. The strategy targets 25-40 trades per year by requiring strong trend (ADX>25), low volatility (BB width < 50th percentile), and price breaking the 20-period EMA with momentum (close > open). Position size is 0.25. Exit on opposite EMA cross or volatility expansion (BB width > 80th percentile).

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
    
    # === 4h data (HTF for trend and volatility) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA20 for trend
    close_4h_series = pd.Series(close_4h)
    ema_20_4h = close_4h_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 4h ADX for trend strength (14-period)
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    
    # +DM and -DM
    up_move = high_4h - np.roll(high_4h, 1)
    down_move = np.roll(low_4h, 1) - low_4h
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed +DM, -DM, TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    atr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / (atr_smooth + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr_smooth + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # 4h Bollinger Bands (20, 2)
    sma_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_width = (bb_upper - bb_lower) / (sma_20 + 1e-10)  # Normalized width
    bb_width_aligned = align_htf_to_ltf(prices, df_4h, bb_width)
    
    # === Entry timing indicators (using 4h close) ===
    # EMA cross signal: close > EMA20 for long, close < EMA20 for short
    ema_cross = close_4h - ema_20_4h
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(bb_width_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        ema_20_val = ema_20_4h_aligned[i]
        adx_val = adx_aligned[i]
        bb_width_val = bb_width_aligned[i]
        ema_cross_val = ema_cross[i]
        
        # Calculate dynamic thresholds for BB width (using historical data up to i)
        bb_width_history = bb_width_aligned[:i+1]
        bb_width_50th = np.percentile(bb_width_history, 50) if len(bb_width_history) > 0 else 0.1
        bb_width_80th = np.percentile(bb_width_history, 80) if len(bb_width_history) > 0 else 0.2
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when EMA cross turns negative OR volatility expands (BB width > 80th percentile)
            if (ema_cross_val < 0) or (bb_width_val > bb_width_80th):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when EMA cross turns positive OR volatility expands
            if (ema_cross_val > 0) or (bb_width_val > bb_width_80th):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: EMA cross positive AND strong trend (ADX>25) AND low volatility (BB width < 50th percentile)
            if (ema_cross_val > 0) and (adx_val > 25) and (bb_width_val < bb_width_50th):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: EMA cross negative AND strong trend (ADX>25) AND low volatility (BB width < 50th percentile)
            elif (ema_cross_val < 0) and (adx_val > 25) and (bb_width_val < bb_width_50th):
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

name = "4h_EMA_Cross_ADX25_BBWidth"
timeframe = "4h"
leverage = 1.0