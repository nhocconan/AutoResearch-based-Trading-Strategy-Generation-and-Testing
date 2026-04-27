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
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h EMA(34) for trend direction
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 4h RSI(14) for momentum
    delta = pd.Series(close_4h).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_aligned = align_htf_to_ltf(prices, df_4h, rsi_14.values)
    
    # Calculate 4h ATR(14) for volatility filter
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(rsi_14_aligned[i]) or
            np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 4h EMA34
        price_above_ema = close[i] > ema_34_4h_aligned[i]
        price_below_ema = close[i] < ema_34_4h_aligned[i]
        
        # Momentum filter: RSI not in extreme territory
        rsi_not_overbought = rsi_14_aligned[i] < 70
        rsi_not_oversold = rsi_14_aligned[i] > 30
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr_14_aligned[i] > np.percentile(atr_14_aligned[:i+1], 20)
        
        # Entry conditions
        long_condition = (price_above_ema and 
                         rsi_not_overbought and 
                         vol_filter)
        
        short_condition = (price_below_ema and 
                          rsi_not_oversold and 
                          vol_filter)
        
        if long_condition and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.20
            position = -1
        # Exit conditions: trend reversal or RSI extreme
        elif position == 1 and (not price_above_ema or rsi_14_aligned[i] > 75):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not price_below_ema or rsi_14_aligned[i] < 25):
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

name = "1h_EMA34_RSI14_VolumeFilter"
timeframe = "1h"
leverage = 1.0