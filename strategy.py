#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + 1w ADX trend filter
# - Long: price breaks above Donchian(20) high + 1d volume > 1.5x 20-period volume average + 1w ADX > 25 and +DI > -DI
# - Short: price breaks below Donchian(20) low + 1d volume > 1.5x 20-period volume average + 1w ADX > 25 and -DI > +DI
# - Exit: ATR trailing stop (highest high since entry - 2.5*ATR for longs, lowest low since entry + 2.5*ATR for shorts)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 20-50 trades/year to stay within fee drag limits while capturing strong trending moves
# - ADX filter ensures we only trade when weekly trend is strong, reducing whipsaws in ranging markets

name = "4h_1d_1w_donchian_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Load 4h data ONCE before loop for Donchian and ATR (MTF rule compliance)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return signals
    
    # Load 1d data ONCE before loop for volume confirmation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Load 1w data ONCE before loop for ADX trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 4h Donchian(20)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR(20) for stoploss
    tr1 = pd.Series(high).rolling(2).max() - pd.Series(low).rolling(2).min()
    tr2 = abs(pd.Series(high).shift(1) - pd.Series(close))
    tr3 = abs(pd.Series(low).shift(1) - pd.Series(close))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_20 = tr.rolling(window=20, min_periods=20).mean().values
    atr_20_aligned = align_htf_to_ltf(prices, df_4h, atr_20)
    
    # Pre-compute 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 1w ADX(14) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1_w = pd.Series(high_1w).rolling(2).max() - pd.Series(low_1w).rolling(2).min()
    tr2_w = abs(pd.Series(high_1w).shift(1) - pd.Series(close_1w))
    tr3_w = abs(pd.Series(low_1w).shift(1) - pd.Series(close_1w))
    tr_w = pd.concat([tr1_w, tr2_w, tr3_w], axis=1).max(axis=1)
    atr_w = tr_w.rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1w).diff()
    down_move = -pd.Series(low_1w).diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_w
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_w
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1w, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1w, minus_di)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(atr_20_aligned[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume average (moderate threshold)
        volume_1d_current = df_1d['volume'].values
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_current)
        vol_confirm = volume_1d_aligned[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Weekly trend filter: ADX > 25 and directional bias
        weekly_adx = adx_aligned[i]
        weekly_plus_di = plus_di_aligned[i]
        weekly_minus_di = minus_di_aligned[i]
        weekly_bullish = weekly_adx > 25 and weekly_plus_di > weekly_minus_di
        weekly_bearish = weekly_adx > 25 and weekly_minus_di > weekly_plus_di
        
        # Donchian breakout conditions
        donchian_breakout_long = close_price > highest_high_20[i]
        donchian_breakout_short = close_price < lowest_low_20[i]
        
        # Entry conditions
        enter_long = donchian_breakout_long and vol_confirm and weekly_bullish
        enter_short = donchian_breakout_short and vol_confirm and weekly_bearish
        
        # Exit conditions: ATR trailing stop
        exit_long = False
        exit_short = False
        
        if position == 1:
            highest_since_entry = max(highest_since_entry, high_price)
            exit_long = close_price < highest_since_entry - 2.5 * atr_20_aligned[i]
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, low_price)
            exit_short = close_price > lowest_since_entry + 2.5 * atr_20_aligned[i]
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            entry_price = close_price
            highest_since_entry = high_price
            lowest_since_entry = low_price
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            entry_price = close_price
            highest_since_entry = high_price
            lowest_since_entry = low_price
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            entry_price = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            entry_price = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals