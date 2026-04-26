#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_RegimeADX
Hypothesis: On 1d timeframe, price breaking Camarilla R1/S1 levels in the direction of 1w EMA50 trend with volume confirmation (>1.5x 20-period MA) and ADX>25 captures high-probability trend continuation moves. The 1w EMA50 acts as a dynamic trend filter, Camarilla levels provide precise entry/exit zones, volume spike confirms institutional participation, and ADX ensures trending market regime. Designed for 7-25 trades/year with discrete sizing (±0.30) and ATR-based trailing stop (2.0x) to minimize fee drag and work in both bull/bear markets with BTC/ETH edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    daily_range = high_1d - low_1d
    camarilla_r1 = close_1d + daily_range * 1.1 / 12
    camarilla_s1 = close_1d - daily_range * 1.1 / 12
    
    # Align Camarilla levels to 1d timeframe (wait for completed 1d bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Load 1w data ONCE before loop for EMA50 trend and ADX regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1w ADX(14) for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range
    tr1 = pd.Series(high_1w).diff().abs()
    tr2 = (pd.Series(high_1w) - pd.Series(close_1w).shift()).abs()
    tr3 = (pd.Series(low_1w) - pd.Series(close_1w).shift()).abs()
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr_1w.ewm(span=14, adjust=False, min_periods=14).mean()
    
    # Calculate Directional Movement
    up_move = pd.Series(high_1w).diff()
    down_move = pd.Series(low_1w).diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Calculate DI and DX
    plus_di = 100 * plus_dm_smooth / atr_1w
    minus_di = 100 * minus_dm_smooth / atr_1w
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1w = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_1w_values = adx_1w.values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w_values)
    
    # 1d ATR(20) for trailing stop
    tr1_d = pd.Series(high).diff().abs()
    tr2_d = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3_d = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr_1d = pd.concat([tr1_d, tr2_d, tr3_d], axis=1).max(axis=1)
    atr_1d = tr_1d.ewm(span=20, adjust=False, min_periods=20).mean()
    atr_1d_values = atr_1d.values
    
    # Volume spike filter: volume > 1.5 * 20-period MA on 1d
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.30
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of EMA (50), ATR (20), volume MA (20) + time for 1d/1w alignment
    start_idx = max(50, 20, 20) + 1  # +1 to ensure 1d bar completion
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        ema_val = ema_50_1w_aligned[i]
        adx_val = adx_1w_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_1d_values[i]
        
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(ema_val) or 
            np.isnan(adx_val) or np.isnan(atr_val) or np.isnan(volume_ma[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: bullish when price > EMA50, bearish when price < EMA50
        trend_bullish = close_val > ema_val
        trend_bearish = close_val < ema_val
        
        # Regime filter: trending market when ADX > 25
        trending_regime = adx_val > 25
        
        # Camarilla breakout conditions: price breaks R1/S1 with trend alignment + volume spike + regime
        long_breakout = close_val > r1_val
        short_breakout = close_val < s1_val
        
        long_entry = trend_bullish and long_breakout and vol_spike and trending_regime
        short_entry = trend_bearish and short_breakout and vol_spike and trending_regime
        
        # Update highest/lowest for trailing stop (ATR-based)
        if position == 1:
            highest_since_long = max(highest_since_long, high_val)
        elif position == -1:
            lowest_since_short = min(lowest_since_short, low_val)
        elif position == 0:
            highest_since_long = 0.0
            lowest_since_short = 0.0
        
        # Exit conditions: ATR-based trailing stoploss
        long_exit = False
        short_exit = False
        if position == 1:
            # Long trailing stop: highest since entry - 2.0 * ATR
            stop_price = highest_since_long - 2.0 * atr_val
            long_exit = close_val < stop_price
        elif position == -1:
            # Short trailing stop: lowest since entry + 2.0 * ATR
            stop_price = lowest_since_short + 2.0 * atr_val
            short_exit = close_val > stop_price
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            highest_since_long = high_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            lowest_since_short = low_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
            highest_since_long = 0.0
        elif short_exit:
            signals[i] = 0.0
            position = 0
            lowest_since_short = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_RegimeADX"
timeframe = "1d"
leverage = 1.0