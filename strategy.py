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
    
    # Get 4h data for higher timeframe context
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h ATR(14) for volatility filter
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Calculate 4h Donchian(20) channels
    donch_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_high_20_aligned = align_htf_to_ltf(prices, df_4h, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_4h, donch_low_20)
    
    # Calculate 4h RSI(14) for overbought/oversold
    delta_4h = pd.Series(close_4h).diff()
    gain_4h = delta_4h.where(delta_4h > 0, 0)
    loss_4h = -delta_4h.where(delta_4h < 0, 0)
    avg_gain_4h = gain_4h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_4h = loss_4h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_4h = avg_gain_4h / avg_loss_4h
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h = rsi_4h.values
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Calculate 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Precompute hour filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_4h_aligned[i]) or 
            np.isnan(donch_high_20_aligned[i]) or 
            np.isnan(donch_low_20_aligned[i]) or
            np.isnan(rsi_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Volatility filter: ATR > 0 (avoid dead markets)
        vol_filter = atr_14_4h_aligned[i] > 0
        
        # Donchian breakout conditions
        breakout_up = close[i] > donch_high_20_aligned[i]
        breakout_down = close[i] < donch_low_20_aligned[i]
        
        # RSI filter: avoid extremes
        rsi_not_overbought = rsi_4h_aligned[i] < 70
        rsi_not_oversold = rsi_4h_aligned[i] > 30
        
        # Long conditions: 4h breakout up + 1d uptrend + RSI not overbought + session + vol
        long_condition = (breakout_up and 
                         price_above_ema and 
                         rsi_not_overbought and 
                         in_session and 
                         vol_filter)
        
        # Short conditions: 4h breakout down + 1d downtrend + RSI not oversold + session + vol
        short_condition = (breakout_down and 
                          price_below_ema and 
                          rsi_not_oversold and 
                          in_session and 
                          vol_filter)
        
        if long_condition and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.20
            position = -1
        # Exit conditions: opposite breakout or trend reversal
        elif position == 1 and (breakout_down or not price_above_ema):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (breakout_up or not price_below_ema):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4hDonchian20_1dEMA50_RSI14_SessionFilter"
timeframe = "1h"
leverage = 1.0