#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Precompute hour filter for 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend context and signal generation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h EMA(34) for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 12h RSI(14)
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14 if i >= 14 else np.nan
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14 if i >= 14 else np.nan
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_12h = 100 - (100 / (1 + rs))
    
    # Calculate 12h ATR(14)
    tr1 = np.abs(high_12h - low_12h)
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = np.full(len(df_12h), np.nan)
    for i in range(14, len(df_12h)):
        atr_12h[i] = np.mean(tr[i-14:i+1])
    
    # Calculate 12h volume moving average (20)
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h indicators to 6h timeframe
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    volume_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    
    # Get 1d data for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    volume_ma_10_1d = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    volume_ma_10_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_10_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(atr_12h_aligned[i]) or np.isnan(volume_ma_20_12h_aligned[i]) or 
            np.isnan(volume_ma_10_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA34
        uptrend = close[i] > ema_34_12h_aligned[i]
        downtrend = close[i] < ema_34_12h_aligned[i]
        
        # Momentum filter: RSI between 40-60 (avoid extremes)
        rsi_mid = (rsi_12h_aligned[i] >= 40) & (rsi_12h_aligned[i] <= 60)
        
        # Volatility filter: current 12h ATR > 0.5 * its 20-period MA (avoid low volatility)
        atr_ma_20_12h = np.full(len(df_12h), np.nan)
        for j in range(34, len(df_12h)):
            if not np.isnan(np.mean(atr_12h[j-19:j+1])):
                atr_ma_20_12h[j] = np.mean(atr_12h[j-19:j+1])
        atr_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_ma_20_12h)
        vol_filter = (not np.isnan(atr_ma_20_12h_aligned[i]) and 
                     atr_12h_aligned[i] > 0.5 * atr_ma_20_12h_aligned[i])
        
        # Volume filter: current volume > 1.3 * 20-period 12h average AND > 1.2 * 10-period 1d average
        vol_spike_12h = volume[i] > 1.3 * volume_ma_20_12h_aligned[i]
        vol_spike_1d = volume[i] > 1.2 * volume_ma_10_1d_aligned[i]
        vol_spike = vol_spike_12h & vol_spike_1d
        
        # Entry conditions: EMA34 trend + RSI mid + volatility + volume spike
        long_entry = uptrend & rsi_mid & vol_filter & vol_spike
        short_entry = downtrend & rsi_mid & vol_filter & vol_spike
        
        # Exit conditions: opposite EMA34 cross or RSI extreme or volatility drop
        long_exit = (~uptrend) | (~rsi_mid) | (~vol_filter)
        short_exit = (~downtrend) | (~rsi_mid) | (~vol_filter)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_1d_ema34_rsi_vol_vol"
timeframe = "6h"
leverage = 1.0