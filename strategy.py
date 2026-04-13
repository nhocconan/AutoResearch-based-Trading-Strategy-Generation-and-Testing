#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for indicator calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR for volatility filtering
    tr_1d = np.maximum(
        high_1d - low_1d,
        np.maximum(
            np.abs(high_1d - np.roll(close_1d, 1)),
            np.abs(low_1d - np.roll(close_1d, 1))
        )
    )
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = np.zeros_like(tr_1d)
    for i in range(len(tr_1d)):
        if i < 14:
            atr_1d[i] = np.mean(tr_1d[:i+1]) if i > 0 else tr_1d[i]
        else:
            atr_1d[i] = 0.93 * atr_1d[i-1] + 0.07 * tr_1d[i]
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily RSI for momentum filter
    delta = np.diff(close_1d, prepend=close_1d[0])
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
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate daily volume moving average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate volatility threshold
    atr_mult = 1.5
    volatility_threshold = atr_1d_aligned * atr_mult
    
    # Calculate price change for volatility filter
    price_change = np.abs(np.diff(close, prepend=close[0]))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(volatility_threshold[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Price volatility filter: avoid choppy markets
        low_volatility = price_change[i] < volatility_threshold[i]
        
        # Trend and momentum filters
        uptrend = close[i] > ema_50_1d_aligned[i]
        strong_momentum = rsi_1d_aligned[i] > 55
        
        downtrend = close[i] < ema_50_1d_aligned[i]
        weak_momentum = rsi_1d_aligned[i] < 45
        
        # Volume confirmation
        volume_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        # Entry conditions
        long_entry = uptrend and strong_momentum and low_volatility and volume_confirm
        short_entry = downtrend and weak_momentum and low_volatility and volume_confirm
        
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

name = "12h_ema50_rsi_vol_filter_v1"
timeframe = "12h"
leverage = 1.0