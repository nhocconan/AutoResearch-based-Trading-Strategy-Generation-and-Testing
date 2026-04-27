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
    
    # Get weekly data for higher timeframe context (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly RSI(14) for trend strength
    delta_1w = pd.Series(close_1w).diff()
    gain_1w = delta_1w.where(delta_1w > 0, 0)
    loss_1w = -delta_1w.where(delta_1w < 0, 0)
    avg_gain_1w = pd.Series(gain_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_1w = pd.Series(loss_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_1w = avg_gain_1w / avg_loss_1w.replace(0, 1e-10)
    rsi_14_1w = 100 - (100 / (1 + rs_1w))
    rsi_14_1w_vals = rsi_14_1w.values
    rsi_14_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_1w_vals)
    
    # Calculate 6-day EMA(20) for trend direction
    ema_20_6d = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_6d_aligned = align_htf_to_ltf(prices, df_1w, ema_20_6d)
    
    # Calculate 6-hour ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 6-hour EMA(50) for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_14_1w_aligned[i]) or 
            np.isnan(ema_20_6d_aligned[i]) or
            np.isnan(ema_50[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: price above/below 6-day EMA on weekly
        weekly_uptrend = close[i] > ema_20_6d_aligned[i]
        weekly_downtrend = close[i] < ema_20_6d_aligned[i]
        
        # Weekly momentum filter: RSI not extreme
        rsi_not_overbought = rsi_14_1w_aligned[i] < 70
        rsi_not_oversold = rsi_14_1w_aligned[i] > 30
        
        # Price trend filter: above/below 6h EMA50
        price_above_ema50 = close[i] > ema_50[i]
        price_below_ema50 = close[i] < ema_50[i]
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_14[i] > 0 and atr_14[i] < np.mean(atr_14[max(0, i-50):i+1]) * 3
        
        # Long conditions: weekly uptrend + RSI not overbought + price above EMA50 + vol filter
        long_condition = (weekly_uptrend and 
                         rsi_not_overbought and 
                         price_above_ema50 and 
                         vol_filter)
        
        # Short conditions: weekly downtrend + RSI not oversold + price below EMA50 + vol filter
        short_condition = (weekly_downtrend and 
                          rsi_not_oversold and 
                          price_below_ema50 and 
                          vol_filter)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal or RSI extreme
        elif position == 1 and (not weekly_uptrend or rsi_14_1w_aligned[i] >= 70):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not weekly_downtrend or rsi_14_1w_aligned[i] <= 30):
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

name = "6h_WeeklyRSI_EMA20_TrendFilter"
timeframe = "6h"
leverage = 1.0