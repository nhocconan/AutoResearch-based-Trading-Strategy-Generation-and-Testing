#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + 1w trend filter
# - Long: price breaks above Donchian(20) high + 1d volume > 2.0x 20-period volume average + 1w close > 1w EMA20
# - Short: price breaks below Donchian(20) low + 1d volume > 2.0x 20-period volume average + 1w close < 1w EMA20
# - Exit: ATR trailing stop (highest high since entry - 3*ATR for longs, lowest low since entry + 3*ATR for shorts)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Works in bull/bear: Donchian captures breakouts; volume confirms participation; weekly trend filter avoids counter-trend trades
# - Target: 20-50 trades/year to stay within fee drag limits while capturing strong moves

name = "4h_1d_1w_donchian_volume_trend_v1"
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
    
    # Load 1w data ONCE before loop for trend filter (MTF rule compliance)
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
    
    # Pre-compute 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(atr_20_aligned[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        # Volume confirmation: 1d volume > 2.0x 20-period volume average (strict threshold)
        volume_1d_current = df_1d['volume'].values
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_current)
        vol_confirm = volume_1d_aligned[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Weekly trend filter: close > EMA20 for bullish, close < EMA20 for bearish
        weekly_close = df_1w['close'].values
        weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
        weekly_bullish = weekly_close_aligned[i] > ema_20_1w_aligned[i]
        weekly_bearish = weekly_close_aligned[i] < ema_20_1w_aligned[i]
        
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
            exit_long = close_price < highest_since_entry - 3.0 * atr_20_aligned[i]
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, low_price)
            exit_short = close_price > lowest_since_entry + 3.0 * atr_20_aligned[i]
        
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