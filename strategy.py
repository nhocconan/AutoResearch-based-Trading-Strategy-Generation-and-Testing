#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions; entries occur on reversals from extremes
# with volume confirmation and trend alignment. Designed for 12h timeframe to reduce trade frequency
# and avoid fee drag. Works in bull (buy oversold in uptrend) and bear (sell overbought in downtrend).
# Targets 15-25 trades/year to stay within fee-efficient range.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Williams %R(14) for overbought/oversold ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Readings: 0 to -20 overbought, -80 to -100 oversold
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R on 12h data
    hh_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = (hh_14 - close_12h) / (hh_14 - ll_14) * -100
    williams_r[hh_14 == ll_14] = -50  # avoid division by zero
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 12h volume ratio for confirmation ===
    volume_12h = df_12h['volume'].values
    vol_ma_10_12h = pd.Series(volume_12h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_12h = volume_12h / vol_ma_10_12h
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ratio_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        wr = williams_r_aligned[i]
        ema_trend_1d = ema_50_1d_aligned[i]
        vol_ratio = vol_ratio_12h_aligned[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            # Stop loss: price closes below entry - 1.5 * ATR (using 12h ATR proxy via price action)
            # Simplified: use 2% of price as volatility proxy for 12h timeframe
            if price < entry_price * 0.98:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Stop loss: price closes above entry + 1.5 * ATR
            if price > entry_price * 1.02:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: Williams %R crosses above -20 (overbought) or trend reverses
            if wr > -20 or price < ema_trend_1d:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -80 (oversold) or trend reverses
            if wr < -80 or price > ema_trend_1d:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R crosses above -80 from below (leaving oversold) with volume, in uptrend
            if (wr > -80 and wr <= -20 and  # just crossed above -80
                vol_ratio > 1.3 and 
                price > ema_trend_1d):
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # SHORT: Williams %R crosses below -20 from above (leaving overbought) with volume, in downtrend
            elif (wr < -20 and wr >= -80 and  # just crossed below -20
                  vol_ratio > 1.3 and 
                  price < ema_trend_1d):
                signals[i] = -0.25
                position = -1
                entry_price = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_WilliamsR_1dEMA50_Volume_Reversal_v1"
timeframe = "12h"
leverage = 1.0