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
    
    # Get weekly data for higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly EMA 21 for trend direction
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate weekly ATR for volatility filter
    tr_1w = np.maximum(high_1w[1:] - low_1w[1:], 
                       np.abs(high_1w[1:] - close_1w[:-1]), 
                       np.abs(low_1w[1:] - close_1w[:-1]))
    tr_1w = np.concatenate([[np.nan], tr_1w])
    atr_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate weekly RSI for overbought/oversold conditions
    delta_1w = pd.Series(close_1w).diff()
    gain_1w = delta_1w.where(delta_1w > 0, 0)
    loss_1w = -delta_1w.where(delta_1w < 0, 0)
    avg_gain_1w = gain_1w.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_1w = loss_1w.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_1w = avg_gain_1w / avg_loss_1w
    rsi_1w = 100 - (100 / (1 + rs_1w))
    rsi_1w = rsi_1w.values
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate daily Donchian channels (20-period)
    # Upper band: highest high of last 20 days
    # Lower band: lowest low of last 20 days
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(atr_1w_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA21
        price_above_ema = close[i] > ema_21_1w_aligned[i]
        price_below_ema = close[i] < ema_21_1w_aligned[i]
        
        # RSI filter: avoid extreme overbought/oversold conditions
        rsi_not_overbought = rsi_1w_aligned[i] < 70
        rsi_not_oversold = rsi_1w_aligned[i] > 30
        
        # Volatility filter: only trade when volatility is reasonable
        vol_filter = atr_1w_aligned[i] > 0
        
        # Donchian breakout conditions
        long_breakout = close[i] > highest_high[i]
        short_breakout = close[i] < lowest_low[i]
        
        # Long conditions: weekly uptrend + RSI not overbought + volatility + Donchian breakout up
        long_condition = (price_above_ema and 
                         rsi_not_overbought and 
                         vol_filter and 
                         long_breakout)
        
        # Short conditions: weekly downtrend + RSI not oversold + volatility + Donchian breakout down
        short_condition = (price_below_ema and 
                          rsi_not_oversold and 
                          vol_filter and 
                          short_breakout)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal or opposite breakout
        elif position == 1 and (not price_above_ema or short_breakout):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not price_below_ema or long_breakout):
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

name = "1w_EMA21_RSI14_Donchian20_Breakout"
timeframe = "1d"
leverage = 1.0