#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ChoppyTrend_Momentum"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 20:
        return np.zeros(n)
    
    # Calculate 20-day EMA on daily close
    close_d = df_d['close'].values
    ema_20_d = pd.Series(close_d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_d_aligned = align_htf_to_ltf(prices, df_d, ema_20_d)
    
    # Get 12-hour data for momentum signal
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate RSI(14) on 12h close
    close_12h = df_12h['close'].values
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Calculate 4h Bollinger Bands for choppiness filter
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20
    # Choppiness: high BB width = choppy market (mean reversion), low BB width = trending
    # We want to trade in trending markets (low BB width)
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    trending_market = bb_width < bb_width_ma  # True when BB width is below average (trending)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_20_d_aligned[i]) or 
            np.isnan(rsi_12h_aligned[i]) or
            np.isnan(trending_market[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_20_val = ema_20_d_aligned[i]
        rsi_val = rsi_12h_aligned[i]
        trending = trending_market[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: Price above daily EMA20 + RSI > 55 + trending market + volume
            if close[i] > ema_20_val and rsi_val > 55 and trending and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: Price below daily EMA20 + RSI < 45 + trending market + volume
            elif close[i] < ema_20_val and rsi_val < 45 and trending and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below EMA20 or RSI < 40
            if close[i] < ema_20_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above EMA20 or RSI > 60
            if close[i] > ema_20_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals