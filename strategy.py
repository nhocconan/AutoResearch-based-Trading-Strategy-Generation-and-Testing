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
    
    # Get weekly data for trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly EMA(10) for trend
    close_1w_series = pd.Series(close_1w)
    ema_10_1w = close_1w_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate weekly ATR(10) for volatility
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = np.full(len(df_1w), np.nan)
    for i in range(10, len(df_1w)):
        atr_1w[i] = np.mean(tr[i-10:i+1])
    
    # Align weekly indicators to daily timeframe
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate daily ATR(10) for position sizing and volatility filter
    tr1_d = np.abs(high - low)
    tr2_d = np.abs(high - np.roll(close, 1))
    tr3_d = np.abs(low - np.roll(close, 1))
    tr1_d[0] = tr2_d[0] = tr3_d[0] = np.nan
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    atr_d = np.full(n, np.nan)
    for i in range(10, n):
        atr_d[i] = np.mean(tr_d[i-10:i+1])
    
    # Calculate daily volume moving average
    vol_s_d = pd.Series(volume)
    vol_ma_10_d = vol_s_d.rolling(window=10, min_periods=10).mean().values
    
    # Calculate daily ATR(10) moving average for volatility filter
    atr_ma_5 = np.full(n, np.nan)
    for i in range(15, n):  # 10 + 4 for 5-period MA
        if not np.isnan(np.mean(atr_d[i-4:i+1])):
            atr_ma_5[i] = np.mean(atr_d[i-4:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema_10_1w_aligned[i]) or np.isnan(atr_1w_aligned[i]) or 
            np.isnan(atr_d[i]) or np.isnan(vol_ma_10_d[i]) or np.isnan(atr_ma_5[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 0.5 * ATR MA(5) to avoid low volatility
        vol_filter = atr_d[i] > 0.5 * atr_ma_5[i]
        
        # Volume filter: volume > 1.3 * 10-period MA
        vol_spike = volume[i] > 1.3 * vol_ma_10_d[i]
        
        # Trend filter: price relative to weekly EMA10
        uptrend = close[i] > ema_10_1w_aligned[i]
        downtrend = close[i] < ema_10_1w_aligned[i]
        
        # Entry conditions: price above/below weekly EMA with volatility and volume filters
        long_entry = uptrend and vol_filter and vol_spike
        short_entry = downtrend and vol_filter and vol_spike
        
        # Exit conditions: price crosses back to weekly EMA10
        long_exit = close[i] < ema_10_1w_aligned[i]
        short_exit = close[i] > ema_10_1w_aligned[i]
        
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

name = "1d_1w_ema10_trend_vol_vol_filter_v1"
timeframe = "1d"
leverage = 1.0