#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_keltner_channel_breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return signals
    
    # Calculate weekly Keltner Channel (20-period EMA, 2.0 ATR multiplier)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # EMA 20
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Bands
    upper_keltner = ema_20 + 2.0 * atr
    lower_keltner = ema_20 - 2.0 * atr
    
    # Align weekly Keltner to daily timeframe
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1w, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1w, lower_keltner)
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX for trend strength (14-period)
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (tr_14 + 1e-10)
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (tr_14 + 1e-10)
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]) or
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        upper = upper_keltner_aligned[i]
        lower = lower_keltner_aligned[i]
        ema = ema_20_aligned[i]
        adx_val = adx_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.8 * vol_ma_20[i]
        
        # Trend filter: ADX > 25
        trending = adx_val > 25
        
        # Entry signals
        long_signal = False
        short_signal = False
        
        # Long: price breaks above weekly upper Keltner with volume and trend
        if price_high > upper and volume_confirmed and trending:
            long_signal = True
        
        # Short: price breaks below weekly lower Keltner with volume and trend
        if price_low < lower and volume_confirmed and trending:
            short_signal = True
        
        # Exit conditions
        # Exit when price returns to weekly EMA
        exit_long = position == 1 and price_close < ema
        exit_short = position == -1 and price_close > ema
        
        # Stop loss: 2x ATR from entry
        # Calculate weekly ATR aligned for stop loss
        atr_1w = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
        atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
        if not np.isnan(atr_1w_aligned[i]):
            atr_val = atr_1w_aligned[i]
            stop_long = position == 1 and price_low < (entry_price - 2.0 * atr_val)
            stop_short = position == -1 and price_high > (entry_price + 2.0 * atr_val)
        else:
            stop_long = False
            stop_short = False
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.30
        elif short_signal and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.30
        elif position == 1 and (exit_long or stop_long):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (exit_short or stop_short):
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Weekly Keltner Channel breakout strategy with volume confirmation and ADX trend filter.
# Enters long when daily price breaks above weekly 20-period Keltner upper band (EMA20 + 2*ATR) with volume > 1.8x average and ADX > 25.
# Enters short when price breaks below weekly lower Keltner band with same conditions.
# Exits when price returns to weekly EMA(20) or 2x ATR stop loss is hit.
# Designed for daily timeframe to target 30-100 total trades over 4 years (7-25/year).
# Works in both bull and bear markets by trading breakouts in either direction with trend filter.