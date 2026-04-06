#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour trading with 4-hour EMA trend filter and 1-day volume confirmation
# Uses EMA(21) on 4h for trend direction, volume spike on 1d for confirmation,
# and RSI(14) on 1h for entry timing. Designed for 1h timeframe to target
# 60-150 trades over 4 years with proper risk management.
# Works in bull/bear via trend filter and mean-reversion entries during pullbacks.

name = "1h_ema4d_vol1d_rsi1h_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4-hour EMA(21) for trend direction
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1-day volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1-hour RSI(14) for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    # Start from warmup period
    start = max(20, 13)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x daily average
        volume_filter = volume[i] > vol_ma_1d_aligned[i] * 1.5
        
        # Trend filter: price above/below 4h EMA
        uptrend = close[i] > ema_4h_aligned[i]
        downtrend = close[i] < ema_4h_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: RSI overbought or stoploss
            if (rsi[i] > 70 or 
                close[i] < entry_price - 2.0 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI oversold or stoploss
            if (rsi[i] < 30 or 
                close[i] > entry_price + 2.0 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: pullbacks in trend with volume confirmation
            if volume_filter:
                # Long: pullback to EMA in uptrend with RSI < 40
                if (uptrend and rsi[i] < 40 and close[i] <= ema_4h_aligned[i] * 1.005):
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                # Short: pullback to EMA in downtrend with RSI > 60
                elif (downtrend and rsi[i] > 60 and close[i] >= ema_4h_aligned[i] * 0.995):
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals