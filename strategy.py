#!/usr/bin/env python3
"""
1h_4h_1d_Trend_Filtered_MeanReversion
Hypothesis: In 1h timeframe, mean reversion works when aligned with higher timeframe trend.
Use 4h trend (EMA21) as filter, 1d volatility (ATR) for dynamic bands, and RSI for entry.
Long when: price < 1h BB lower band + RSI < 30 + 4h EMA21 upward slope
Short when: price > 1h BB upper band + RSI > 70 + 4h EMA21 downward slope
Exit when price crosses 1h VWAP or RSI reverts to 50.
Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h trend filter - EMA21 slope
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_slope = np.diff(ema_4h, prepend=ema_4h[0])
    ema_4h_up = ema_4h_slope > 0
    ema_4h_down = ema_4h_slope < 0
    
    ema_4h_up_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_up.astype(float))
    ema_4h_down_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_down.astype(float))
    
    # 1d volatility - ATR for dynamic bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan
    
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1h indicators
    # RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_lower = sma - (bb_std_dev * bb_std)
    bb_upper = sma + (bb_std_dev * bb_std)
    
    # VWAP for exit
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_4h_up_aligned[i]) or 
            np.isnan(ema_4h_down_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(bb_lower[i]) or 
            np.isnan(bb_upper[i]) or 
            np.isnan(vwap[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_cond = (close[i] < bb_lower[i] and 
                     rsi[i] < 30 and 
                     ema_4h_up_aligned[i] > 0.5)
        
        short_cond = (close[i] > bb_upper[i] and 
                      rsi[i] > 70 and 
                      ema_4h_down_aligned[i] > 0.5)
        
        # Exit conditions
        exit_long = position == 1 and (close[i] > vwap[i] or rsi[i] > 50)
        exit_short = position == -1 and (close[i] < vwap[i] or rsi[i] < 50)
        
        # Execute signals
        if long_cond and position != 1:
            position = 1
            signals[i] = position_size
        elif short_cond and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_Trend_Filtered_MeanReversion"
timeframe = "1h"
leverage = 1.0