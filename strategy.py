#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR trailing stop
# - Long: price breaks above Donchian upper (20-period high) with volume > 1.5x 20-period average volume
# - Short: price breaks below Donchian lower (20-period low) with volume > 1.5x 20-period average volume
# - Exit: trailing stop at 2.5x ATR(14) from highest high (long) or lowest low (short)
# - Uses 12h EMA(21) as trend filter: only long when price > EMA12h, only short when price < EMA12h
# - Works in bull markets via breakouts, in bear markets via short breakdowns with trend filter
# - Target: 15-40 trades/year (60-160 total over 4 years) to stay within fee drag limits

name = "4h_12h_donchian_volume_trend_v1"
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
    
    # Load 12h data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return signals
    
    # Pre-compute 12h EMA(21) for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR(14) for trailing stop
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.concatenate([[close[0]], close[:-1]]) - high)
    tr3 = np.abs(np.concatenate([[close[0]], close[:-1]]) - low)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or
            np.isnan(ema_12h_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Donchian levels
        upper_channel = high_max_20[i]
        lower_channel = low_min_20[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Trend filter from 12h EMA
        price_above_ema = close_price > ema_12h_aligned[i]
        price_below_ema = close_price < ema_12h_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above upper channel with volume and trend filter
        if close_price > upper_channel and vol_confirm and price_above_ema:
            enter_long = True
        
        # Short breakout: price breaks below lower channel with volume and trend filter
        if close_price < lower_channel and vol_confirm and price_below_ema:
            enter_short = True
        
        # Exit conditions: trailing stop at 2.5x ATR
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Update highest high for trailing stop
            highest_high = max(highest_high, high_price)
            exit_long = close_price < (highest_high - 2.5 * atr[i])
        elif position == -1:
            # Update lowest low for trailing stop
            lowest_low = min(lowest_low, low_price)
            exit_short = close_price > (lowest_low + 2.5 * atr[i])
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            entry_price = close_price
            highest_high = high_price
            lowest_low = low_price
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            entry_price = close_price
            highest_high = high_price
            lowest_low = low_price
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            highest_high = 0.0
            lowest_low = 0.0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            highest_high = 0.0
            lowest_low = 0.0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals