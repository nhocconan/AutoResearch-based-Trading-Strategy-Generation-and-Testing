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
    
    # Get 4h data for trend and volatility
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h EMA(34) for trend filter
    ema34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # 4h ATR(14) for volatility filter
    tr1_4h = df_4h['high'].values - df_4h['low'].values
    tr2_4h = np.abs(df_4h['high'].values - np.roll(df_4h['close'].values, 1))
    tr3_4h = np.abs(df_4h['low'].values - np.roll(df_4h['close'].values, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_4h_raw = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h_raw)
    
    # Get 1d data for higher timeframe trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA(50) for higher timeframe trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1h RSI(14) for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup
    start_idx = max(34, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(atr_4h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema34_4h_aligned[i]
        atr_4h_val = atr_4h_aligned[i]
        ema50_trend = ema50_1d_aligned[i]
        rsi_val = rsi[i]
        in_session = session_filter[i]
        
        # Volatility filter: 4h ATR > 0.3 * price (avoid low volatility chop)
        vol_filter = atr_4h_val > (close[i] * 0.003)
        
        if position == 0:
            # Long: price above 4h EMA, above 1d EMA, RSI not overbought, in session, volatility filter
            if (close[i] > ema_trend and close[i] > ema50_trend and 
                rsi_val < 70 and vol_filter and in_session):
                signals[i] = size
                position = 1
            # Short: price below 4h EMA, below 1d EMA, RSI not oversold, in session, volatility filter
            elif (close[i] < ema_trend and close[i] < ema50_trend and 
                  rsi_val > 30 and vol_filter and in_session):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 4h EMA or RSI overbought
            if close[i] < ema_trend or rsi_val > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 4h EMA or RSI oversold
            if close[i] > ema_trend or rsi_val < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_EMA34_EMA50_RSI_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0