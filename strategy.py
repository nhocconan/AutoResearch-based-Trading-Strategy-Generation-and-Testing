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
    
    # Get weekly data once for HTF context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w indicators
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # 1w EMA(20) for trend
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1w RSI(14) for momentum
    delta = pd.Series(close_1w).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean().values
    avg_loss = loss.rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1w ATR(14) for volatility
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to daily timeframe
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    atr_14_aligned = align_htf_to_ltf(prices, df_1w, atr_14)
    
    # Day filter: Monday to Friday (weekday 0-4)
    weekdays = pd.DatetimeIndex(prices['open_time']).weekday
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekday filter: only trade Monday-Friday
        weekday = weekdays[i]
        is_weekday = weekday < 5  # Monday=0, Friday=4
        
        if not is_weekday:
            # Weekend: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA20
        trend_up = close[i] > ema_20_aligned[i]
        trend_down = close[i] < ema_20_aligned[i]
        
        # Momentum filter: RSI in neutral range (avoid extremes)
        rsi_neutral = (rsi_aligned[i] >= 40) & (rsi_aligned[i] <= 60)
        
        # Volume filter: above average volume
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_filter = volume[i] > vol_ma[i]
        
        # Entry conditions - selective to reduce trades
        long_entry = trend_up and rsi_neutral and vol_filter
        short_entry = trend_down and rsi_neutral and vol_filter
        
        # Exit conditions: opposite conditions or volatility spike
        atr_ma = pd.Series(atr_14_aligned).rolling(window=10, min_periods=10).mean().values
        vol_spike = atr_14_aligned[i] > 2.5 * atr_ma[i]
        long_exit = not trend_up or not rsi_neutral or vol_spike
        short_exit = not trend_down or not rsi_neutral or vol_spike
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_EMA20_RSI_Volume_Weekday"
timeframe = "1d"
leverage = 1.0