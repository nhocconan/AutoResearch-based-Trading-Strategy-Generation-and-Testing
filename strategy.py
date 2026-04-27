#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for higher timeframe context (1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA(34) for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily RSI(14) for momentum
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate 12h Bollinger Bands (20,2) for volatility and mean reversion
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    bb_middle = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_upper_aligned = align_htf_to_ltf(prices, df_12h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_12h, bb_lower)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(rsi_14_1d_aligned[i]) or
            np.isnan(bb_upper_aligned[i]) or
            np.isnan(bb_lower_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # RSI filter: avoid extreme overbought/oversold
        rsi_not_overbought = rsi_14_1d_aligned[i] < 70
        rsi_not_oversold = rsi_14_1d_aligned[i] > 30
        
        # Mean reversion signals: price touches Bollinger Bands
        touch_upper = close[i] >= bb_upper_aligned[i]
        touch_lower = close[i] <= bb_lower_aligned[i]
        
        # Long conditions: bullish trend + RSI not overbought + touch lower BB
        long_condition = (price_above_ema and 
                         rsi_not_overbought and 
                         touch_lower)
        
        # Short conditions: bearish trend + RSI not oversold + touch upper BB
        short_condition = (price_below_ema and 
                          rsi_not_oversold and 
                          touch_upper)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal or RSI extreme
        elif position == 1 and (not price_above_ema or rsi_14_1d_aligned[i] >= 70):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not price_below_ema or rsi_14_1d_aligned[i] <= 30):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_EMA34_RSI14_BBMeanReversion"
timeframe = "12h"
leverage = 1.0