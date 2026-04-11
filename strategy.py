#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR trailing stop
# - Long: price breaks above 20-period Donchian high with volume > 1.5x 20-period avg volume
# - Short: price breaks below 20-period Donchian low with volume > 1.5x 20-period avg volume
# - Exit: trailing stop at 2.5 * ATR(14) from highest high (long) or lowest low (short)
# - Uses 1d EMA(50) as trend filter: only long when price > EMA50, short when price < EMA50
# - Works in both bull and bear markets by combining breakouts with trend filter and volatility stops
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits

name = "4h_1d_donchian_volume_trend_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_high = 0.0
    lowest_low = 0.0
    
    # Load 1d data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 4h Donchian channels (20-period)
    lookback = 20
    highest_high_20 = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low_20 = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 4h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(atr_14[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Donchian levels
        donch_high = highest_high_20[i]
        donch_low = lowest_low_20[i]
        
        # Trend filter from 1d EMA50
        ema50 = ema_50_1d_aligned[i]
        
        # ATR for trailing stop
        atr = atr_14[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above Donchian high with volume and trend filter
        if close_price > donch_high and vol_confirm and close_price > ema50:
            enter_long = True
        
        # Short breakout: price breaks below Donchian low with volume and trend filter
        if close_price < donch_low and vol_confirm and close_price < ema50:
            enter_short = True
        
        # Exit conditions: trailing stop
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Update highest high for trailing stop
            highest_high = max(highest_high, high_price)
            # Exit long if price drops to highest_high - 2.5 * ATR
            exit_long = close_price <= (highest_high - 2.5 * atr)
        elif position == -1:
            # Update lowest low for trailing stop
            lowest_low = min(lowest_low, low_price)
            # Exit short if price rises to lowest_low + 2.5 * ATR
            exit_short = close_price >= (lowest_low + 2.5 * atr)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            entry_price = close_price
            highest_high = high_price
            lowest_low = low_price
            signals[i] = 0.30
        elif enter_short and position != -1:
            position = -1
            entry_price = close_price
            highest_high = high_price
            lowest_low = low_price
            signals[i] = -0.30
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals