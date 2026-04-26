#!/usr/bin/env python3
"""
6h_ADX_DI_Cross_12hTrend_VolumeFilter
Hypothesis: On 6h timeframe, enter long when +DI crosses above -DI with ADX>25 (trend strength), aligned with 12h EMA50 uptrend and volume confirmation (>1.5x 20-bar MA). Enter short when -DI crosses above +DI with ADX>25, aligned with 12h EMA50 downtrend and volume confirmation. Uses ADX to filter weak trends and DI crossovers for precise entries. Volume confirmation reduces false signals. Designed for 15-25 trades/year (60-100 total over 4 years) to avoid fee drag. Works in bull markets via long entries and bear markets via short entries, both filtered by 12h trend and ADX strength.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # ADX and DI calculation on 6h data
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed DM and TR
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations (14 for ADX/DI, 20 for volume, 50 for 12h EMA)
    start_idx = max(14, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        ema_12h_val = ema_50_12h_aligned[i]
        adx_val = adx[i]
        plus_di_val = plus_di[i]
        minus_di_val = minus_di[i]
        vol_conf = volume_confirm[i]
        
        # Trend filters
        bullish_12h = close_val > ema_12h_val
        bearish_12h = close_val < ema_12h_val
        
        # ADX trend strength filter
        strong_trend = adx_val > 25
        
        # DI crossover conditions
        # Long: +DI crosses above -DI (current +DI > -DI and previous +DI <= previous -DI)
        # Short: -DI crosses above +DI (current -DI > +DI and previous -DI <= previous +DI)
        if i > 0:
            prev_plus_di = plus_di[i-1]
            prev_minus_di = minus_di[i-1]
            bullish_cross = (plus_di_val > minus_di_val) and (prev_plus_di <= prev_minus_di)
            bearish_cross = (minus_di_val > plus_di_val) and (prev_minus_di <= prev_plus_di)
        else:
            bullish_cross = False
            bearish_cross = False
        
        # Entry conditions: DI crossover in trend direction with ADX>25 and volume confirmation
        long_entry = bullish_cross and bullish_12h and strong_trend and vol_conf
        short_entry = bearish_cross and bearish_12h and strong_trend and vol_conf
        
        # Exit conditions: opposite DI crossover or trend weakening (ADX<20) or trend reversal
        exit_long = bearish_cross or (adx_val < 20) or not bullish_12h
        exit_short = bullish_cross or (adx_val < 20) or not bearish_12h
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_ADX_DI_Cross_12hTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0