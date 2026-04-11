#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_keltner_channel_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return signals
    
    # Calculate weekly Keltner Channel: EMA20 ± ATR(10)*2
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 20-period EMA of close
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # True Range for ATR
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Channel bands
    upper_keltner = ema_20 + 2 * atr_10
    lower_keltner = ema_20 - 2 * atr_10
    
    # Shift by 1 to use only completed weekly bars
    upper_keltner = np.roll(upper_keltner, 1)
    lower_keltner = np.roll(lower_keltner, 1)
    ema_20 = np.roll(ema_20, 1)
    upper_keltner[0] = np.nan
    lower_keltner[0] = np.nan
    ema_20[0] = np.nan
    
    # Align weekly indicators to 12h timeframe
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1w, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1w, lower_keltner)
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Calculate 12h ATR for stop loss
    tr_12h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr_12h = pd.Series(tr_12h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]) or
            np.isnan(ema_20_aligned[i]) or np.isnan(atr_12h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr_12h[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Long conditions: Close > Upper Keltner with volume
        long_signal = volume_confirmed and (price_close > upper_keltner_aligned[i])
        
        # Short conditions: Close < Lower Keltner with volume
        short_signal = volume_confirmed and (price_close < lower_keltner_aligned[i])
        
        # Exit when price crosses back to EMA20 (mean reversion)
        exit_long = position == 1 and price_close < ema_20_aligned[i]
        exit_short = position == -1 and price_close > ema_20_aligned[i]
        
        # Stop loss: 2 * ATR
        stop_long = position == 1 and price_low < (entry_price - 2 * atr)
        stop_short = position == -1 and price_high > (entry_price + 2 * atr)
        
        # Track entry price for stop loss
        if 'entry_price' not in locals():
            entry_price = 0
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif (position == 1 and (exit_long or stop_long)) or (position == -1 and (exit_short or stop_short)):
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Weekly Keltner Channel breakout with volume confirmation on 12h.
# Uses weekly EMA20 ± ATR(10)*2 as dynamic support/resistance. Enters long when price
# breaks above upper channel with volume confirmation (>1.5x average volume).
# Enters short when price breaks below lower channel with volume confirmation.
# Exits when price crosses back to the weekly EMA20 (mean reversion) or hits
# 2x ATR stop loss. Works in both bull and bear markets by capturing breakouts
# from volatility contractions. Weekly timeframe filters noise, 12h provides
# timely execution. Target: 50-150 total trades over 4 years (12-37/year) to
# minimize fee drag on 12h timeframe. Keltner Channels adapt to volatility,
# providing better breakout signals than fixed channels. Volume confirmation
# ensures institutional participation. EMA20 exit captures mean reversion
# after overextended moves.