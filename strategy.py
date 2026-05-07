#!/usr/bin/env python3
name = "6h_200EMA_RSI_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for 200EMA and RSI (weekly for higher timeframe context)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 200 EMA on daily close
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 14-period RSI on daily close
    delta = pd.Series(df_1d['close'].values).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], 100)  # Handle division by zero
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d = rsi_14_1d.fillna(50).values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Weekly trend filter: 50 EMA on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 6h RSI for entry timing (14-period)
    delta_6h = pd.Series(close).diff()
    gain_6h = delta_6h.clip(lower=0)
    loss_6h = -delta_6h.clip(upper=0)
    avg_gain_6h = gain_6h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_6h = loss_6h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_6h = avg_gain_6h / avg_loss_6h
    rs_6h = rs_6h.replace([np.inf, -np.inf], 100)
    rsi_14_6h = 100 - (100 / (1 + rs_6h))
    rsi_14_6h = rsi_14_6h.fillna(50).values
    
    # Volatility filter: ATR > 0.3% of price
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    vol_filter = atr > 0.003 * close
    
    # Session filter: 08:00 - 20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 14)  # Ensure 200EMA and RSI data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN or invalid
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(rsi_14_6h[i]) or
            np.isnan(vol_filter[i]) or not vol_filter[i] or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above daily 200EMA (bullish regime), 
            # weekly 50EMA confirms uptrend, daily RSI oversold (<30), 6h RSI crosses above 30
            if (close[i] > ema_200_1d_aligned[i] and
                ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] and  # Weekly EMA rising
                rsi_14_1d_aligned[i] < 30 and
                rsi_14_6h[i] > 30 and rsi_14_6h[i-1] <= 30):
                signals[i] = 0.25
                position = 1
            # Short: Price below daily 200EMA (bearish regime),
            # weekly 50EMA confirms downtrend, daily RSI overbought (>70), 6h RSI crosses below 70
            elif (close[i] < ema_200_1d_aligned[i] and
                  ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] and  # Weekly EMA falling
                  rsi_14_1d_aligned[i] > 70 and
                  rsi_14_6h[i] < 70 and rsi_14_6h[i-1] >= 70):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: RSI mean reversion - exit when RSI returns to neutral zone (40-60)
            if 40 <= rsi_14_6h[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals