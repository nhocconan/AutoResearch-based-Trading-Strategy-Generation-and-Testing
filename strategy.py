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
    
    # Get weekly data for trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR for volatility filter
    tr_1w = np.maximum(
        high_1w - low_1w,
        np.maximum(
            np.abs(high_1w - np.roll(close_1w, 1)),
            np.abs(low_1w - np.roll(close_1w, 1))
        )
    )
    tr_1w[0] = high_1w[0] - low_1w[0]
    atr_1w = np.zeros_like(tr_1w)
    for i in range(len(tr_1w)):
        if i < 14:
            atr_1w[i] = np.mean(tr_1w[:i+1]) if i > 0 else tr_1w[i]
        else:
            atr_1w[i] = 0.93 * atr_1w[i-1] + 0.07 * tr_1w[i]
    
    # Calculate weekly EMA200 for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate weekly RSI for momentum filter
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    for i in range(len(gain)):
        if i < 14:
            avg_gain[i] = np.mean(gain[:i+1]) if i > 0 else gain[i]
            avg_loss[i] = np.mean(loss[:i+1]) if i > 0 else loss[i]
        else:
            avg_gain[i] = 0.92 * avg_gain[i-1] + 0.08 * gain[i]
            avg_loss[i] = 0.92 * avg_loss[i-1] + 0.08 * loss[i]
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Align indicators to daily timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate 20-week ATR multiple for volatility filter
    atr_mult = 1.5
    volatility_threshold = atr_1w_aligned * atr_mult
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(rsi_1w_aligned[i]) or
            np.isnan(volatility_threshold[i])):
            signals[i] = 0.0
            continue
        
        # Price volatility filter: avoid choppy markets
        price_change = np.abs(close[i] - close[i-1])
        low_volatility = price_change < volatility_threshold[i]
        
        # Trend and momentum filters
        uptrend = close[i] > ema_200_1w_aligned[i]
        strong_momentum = rsi_1w_aligned[i] > 50
        
        downtrend = close[i] < ema_200_1w_aligned[i]
        weak_momentum = rsi_1w_aligned[i] < 50
        
        # Entry conditions
        long_entry = uptrend and strong_momentum and low_volatility
        short_entry = downtrend and weak_momentum and low_volatility
        
        # Exit conditions: trend reversal
        exit_long = position == 1 and (not uptrend or not strong_momentum)
        exit_short = position == -1 and (not downtrend or not weak_momentum)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
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

name = "1d_weekly_ema200_rsi_momentum_filter_v1"
timeframe = "1d"
leverage = 1.0