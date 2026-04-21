#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h EMA21 for trend filter
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 1d RSI(14) for overbought/oversold
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # 1h volume confirmation: volume / 20-period average volume
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # Session filter: 8-20 UTC (pre-compute)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if indicators not ready
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_21_4h_aligned[i]
        rsi = rsi_14_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 1.5  # Volume must be above average
        
        if position == 0:
            # Enter long: price above EMA21, RSI oversold, volume spike
            if (price_close > ema_trend and 
                rsi < 30 and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.20
                position = 1
            # Enter short: price below EMA21, RSI overbought, volume spike
            elif (price_close < ema_trend and 
                  rsi > 70 and 
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit: reverse condition or RSI mean reversion
            if position == 1 and (price_close < ema_trend or rsi > 50):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > ema_trend or rsi < 50):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_EMA21_RSI_Volume_Session"
timeframe = "1h"
leverage = 1.0