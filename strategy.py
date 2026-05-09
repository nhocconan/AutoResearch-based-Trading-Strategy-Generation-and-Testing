# 12h_SMA20_RSI25_VolumeSpike
# Hypothesis: Combines long-term trend (12h SMA20), short-term momentum (12h RSI < 25 for oversold, >75 for overbought), and volume confirmation.
# Uses daily ADX > 25 as a regime filter to ensure trading only in trending markets.
# Designed to capture oversold bounces in uptrends and overbought pullbacks in downtrends.
# Target: 15-35 trades/year to stay within fee-efficient range.
# Works in bull markets by buying oversold dips in uptrends.
# Works in bear markets by selling overbought rallies in downtrends.
# Uses 12h timeframe as specified.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_SMA20_RSI25_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # Calculate 14-period ADX for daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0]) * -1
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (1 - alpha) * atr[i-1] + alpha * tr[i]
    
    # Avoid division by zero
    atr_safe = np.where(atr == 0, 1e-10, atr)
    
    # Smoothed +DM and -DM
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    plus_dm_smooth[0] = plus_dm[0]
    minus_dm_smooth[0] = minus_dm[0]
    for i in range(1, len(plus_dm)):
        plus_dm_smooth[i] = (1 - alpha) * plus_dm_smooth[i-1] + alpha * plus_dm[i]
        minus_dm_smooth[i] = (1 - alpha) * minus_dm_smooth[i-1] + alpha * minus_dm[i]
    
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_smooth / atr_safe
    minus_di = 100 * minus_dm_smooth / atr_safe
    
    # Calculate DX and ADX
    dx = np.where((plus_di + minus_di) > 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = np.zeros_like(dx)
    adx[0] = dx[0]
    for i in range(1, len(dx)):
        adx[i] = (1 - alpha) * adx[i-1] + alpha * dx[i]
    
    # Calculate 20-period SMA for 12h timeframe
    sma20 = np.full_like(close, np.nan)
    for i in range(len(close)):
        if i < 19:
            sma20[i] = np.nan
        else:
            sma20[i] = np.mean(close[i-19:i+1])
    
    # Calculate 14-period RSI for 12h timeframe
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, len(gain)):
        avg_gain[i] = (1 - 1/rsi_period) * avg_gain[i-1] + (1/rsi_period) * gain[i]
        avg_loss[i] = (1 - 1/rsi_period) * avg_loss[i-1] + (1/rsi_period) * loss[i]
    
    rs = np.where(avg_loss == 0, 100, avg_gain / np.where(avg_loss == 0, 1, avg_loss))
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.8x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.8 * vol_ema20)
    
    # Align 1d ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Need both SMA20 and RSI periods
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(adx_aligned[i]) or np.isnan(sma20[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price > SMA20 (uptrend) + RSI < 25 (oversold) + ADX > 25 + volume spike
            if (price > sma20[i] and rsi[i] < 25 and adx_aligned[i] > 25 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < SMA20 (downtrend) + RSI > 75 (overbought) + ADX > 25 + volume spike
            elif (price < sma20[i] and rsi[i] > 75 and adx_aligned[i] > 25 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < SMA20 or RSI > 60 (overbought exit) or ADX drops below 20
            if price < sma20[i] or rsi[i] > 60 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > SMA20 or RSI < 40 (oversold exit) or ADX drops below 20
            if price > sma20[i] or rsi[i] < 40 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals