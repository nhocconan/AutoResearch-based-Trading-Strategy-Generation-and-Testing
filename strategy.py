#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX25 regime filter and EMA50 trend filter
# Elder Ray measures bull/bear power relative to EMA13; ADX>25 confirms trending market
# Only take long when Bull Power > 0 and ADX>25 and price>EMA50, short when Bear Power < 0 and ADX>25 and price<EMA50
# Designed for ~15-25 trades/year to minimize fee drag while capturing strong trends in both bull and bear markets
# Works in bull via long signals, works in bear via short signals when ADX confirms downtrend

name = "6h_ElderRay_1dADX25_EMA50Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX25 and EMA50 trend filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # need enough for ADX and EMA
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(high_1d - np.roll(high_1d, 1))
    down_move = pd.Series(np.roll(low_1d, 1) - low_1d)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    plus_di = 100 * plus_dm_smooth / atr_1d
    minus_di = 100 * minus_dm_smooth / atr_1d
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = 50  # warmup for ADX and EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_13[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema13 = ema_13[i]
        curr_adx = adx_1d_aligned[i]
        curr_ema50 = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        
        # Calculate Elder Ray components
        bull_power = curr_high - curr_ema13  # Bull Power = High - EMA13
        bear_power = curr_low - curr_ema13   # Bear Power = Low - EMA13
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or ADX < 20 (trend weakening) or Bear Power > 0 (momentum shift)
            if (curr_close < entry_price - 2.5 * atr_at_entry or 
                curr_adx < 20 or 
                bear_power > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or ADX < 20 (trend weakening) or Bull Power < 0 (momentum shift)
            if (curr_close > entry_price + 2.5 * atr_at_entry or 
                curr_adx < 20 or 
                bull_power < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # ADX filter: only trade when ADX > 25 (strong trend)
            # EMA50 filter: only trade in direction of daily trend
            strong_trend = curr_adx > 25
            above_ema50 = curr_close > curr_ema50
            below_ema50 = curr_close < curr_ema50
            
            # Long when Bull Power > 0, strong uptrend (ADX>25), and price above daily EMA50
            if bull_power > 0 and strong_trend and above_ema50:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
            # Short when Bear Power < 0, strong downtrend (ADX>25), and price below daily EMA50
            elif bear_power < 0 and strong_trend and below_ema50:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals