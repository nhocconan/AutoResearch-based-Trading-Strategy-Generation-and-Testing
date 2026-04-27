#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly EMA(13) for trend direction
    ema_13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_13_1w)
    
    # Calculate weekly RSI(14) for overbought/oversold conditions
    delta = pd.Series(close_1w).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1w = (100 - (100 / (1 + rs))).values
    rsi_14_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    
    # Calculate daily ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = high_1d[0] - close_1d[0]
    tr3[0] = low_1d[0] - close_1d[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_13_1w_aligned[i]) or 
            np.isnan(rsi_14_1w_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA13
        price_above_ema = close[i] > ema_13_1w_aligned[i]
        price_below_ema = close[i] < ema_13_1w_aligned[i]
        
        # RSI filter: avoid overbought/oversold extremes
        rsi_not_overbought = rsi_14_1w_aligned[i] < 70
        rsi_not_oversold = rsi_14_1w_aligned[i] > 30
        
        # Volatility filter: current price change less than 2x ATR
        price_change = np.abs(close[i] - close[i-1]) if i > 0 else 0
        vol_filter = price_change < (2.0 * atr_14_1d_aligned[i])
        
        # Long conditions: bullish trend + not overbought + volatility filter
        long_condition = (price_above_ema and 
                         rsi_not_overbought and 
                         vol_filter)
        
        # Short conditions: bearish trend + not oversold + volatility filter
        short_condition = (price_below_ema and 
                          rsi_not_oversold and 
                          vol_filter)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal or RSI extreme
        elif position == 1 and (not price_above_ema or rsi_14_1w_aligned[i] >= 70):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not price_below_ema or rsi_14_1w_aligned[i] <= 30):
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

name = "1d_EMA13_1wRSI14_VolatilityFilter"
timeframe = "1d"
leverage = 1.0