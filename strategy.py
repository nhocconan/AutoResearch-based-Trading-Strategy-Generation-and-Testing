#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 1h momentum with 4h trend filter and 1d volatility filter
    # Uses RSI(2) for mean reversion entries in direction of 4h EMA(50) trend
    # Only trades when 1d volatility is expanding (avoids chop)
    # Target: 15-35 trades/year per symbol
    
    # Session filter: 8:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # 4h EMA(50) for trend
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d ATR(14) for volatility
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr[i-14:i+1])
    
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d ATR MA(20) for volatility expansion filter
    atr_ma_20_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20_1d)
    
    # Volatility filter: current ATR > 1.2 * its 20-period average (expanding volatility)
    vol_expansion = (atr_1d_aligned > 1.2 * atr_ma_20_1d_aligned) & ~(np.isnan(atr_1d_aligned) | np.isnan(atr_ma_20_1d_aligned))
    
    # 1h RSI(2) for mean reversion entry
    rsi_period = 2
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(gain, np.nan, dtype=float)
    avg_loss = np.full_like(loss, np.nan, dtype=float)
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    
    for i in range(rsi_period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_2 = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(rsi_2[i]) or 
            np.isnan(vol_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 4h EMA(50)
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        # RSI(2) extreme readings for mean reversion
        rsi_oversold = rsi_2[i] < 10
        rsi_overbought = rsi_2[i] > 90
        
        # Entry conditions: mean reversion in direction of trend
        long_entry = uptrend and rsi_oversold and vol_expansion[i]
        short_entry = downtrend and rsi_overbought and vol_expansion[i]
        
        # Exit conditions: RSI returns to neutral or trend changes
        long_exit = (rsi_2[i] > 50) or (not uptrend)
        short_exit = (rsi_2[i] < 50) or (not downtrend)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_rsi2_trend_vol_filter_v1"
timeframe = "1h"
leverage = 1.0