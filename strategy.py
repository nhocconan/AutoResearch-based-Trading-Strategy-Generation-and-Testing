#!/usr/bin/env python3
# 1h_4h_1d_camarilla_breakout_volatility_v1
# Strategy: 1-hour Camarilla breakout with 4-hour trend filter and 1-day volatility filter
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: Uses 1-hour price action for entry timing, filtered by 4-hour trend direction
# and 1-day volatility regime (low volatility = breakout prone). Designed to capture
# breakouts during low volatility periods with trend alignment, reducing false signals
# in choppy markets. Target: 15-35 trades/year (~60-140 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_breakout_volatility_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h OHLC for trend (using close for EMA)
    close_4h = df_4h['close'].values
    
    # 1d OHLC for ATR calculation (volatility filter)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 4h EMA20 for trend filter
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1-day ATR(14) for volatility filter
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 4h and 1d data to 1h timeframe
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1-hour ATR for position sizing/context
    hl = high - low
    hc = np.abs(high - np.roll(close, 1))
    lc = np.abs(low - np.roll(close, 1))
    tr_1h = np.maximum(hl, np.maximum(hc, lc))
    atr_10_1h = pd.Series(tr_1h).rolling(window=10, min_periods=10).mean().values
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(atr_10_1h[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        price_close = close[i]
        
        # Trend filter: price vs 4h EMA20
        uptrend_4h = price_close > ema_20_4h_aligned[i]
        downtrend_4h = price_close < ema_20_4h_aligned[i]
        
        # Volatility filter: low volatility environment (ATR below 20-period MA)
        vol_filter = atr_10_1h[i] < (0.8 * atr_14_1d_aligned[i])
        
        # Calculate 1-hour ATR-based bands for breakout detection
        atr_mult = 1.5
        upper_band = close[i-1] + (atr_mult * atr_10_1h[i-1])
        lower_band = close[i-1] - (atr_mult * atr_10_1h[i-1])
        
        # Breakout signals
        breakout_up = price_close > upper_band
        breakdown_down = price_close < lower_band
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: Break above upper band with volume in uptrend AND low volatility
        long_signal = breakout_up and vol_confirmed and uptrend_4h and vol_filter
        
        # Short: Break below lower band with volume in downtrend AND low volatility
        short_signal = breakdown_down and vol_confirmed and downtrend_4h and vol_filter
        
        # Exit when price returns to the previous close or opposite band
        exit_long = position == 1 and price_close < close[i-1]
        exit_short = position == -1 and price_close > close[i-1]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals