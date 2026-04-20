#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_Momentum_Trend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # === 4h: Trend filter (EMA50) ===
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # === 1d: Momentum filter (RSI14) ===
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_loss / np.where(avg_gain > 0, avg_gain, np.nan)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 1h: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip outside session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema50_4h_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_val) or np.isnan(rsi_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Uptrend + bullish momentum + volume confirmation
            if (close_val > ema_val and          # Price above 4h EMA50 (uptrend)
                40 < rsi_val < 70 and            # 1d RSI in bullish range (not overbought)
                vol_ratio_val > 1.5):            # Volume confirmation
                signals[i] = 0.20
                position = 1
            # Short: Downtrend + bearish momentum + volume confirmation
            elif (close_val < ema_val and        # Price below 4h EMA50 (downtrend)
                  30 < rsi_val < 60 and          # 1d RSI in bearish range (not oversold)
                  vol_ratio_val > 1.5):          # Volume confirmation
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: trend breakdown or momentum fade
            if (close_val < ema_val or           # Price below 4h EMA50
                rsi_val > 75 or                  # 1d RSI overbought
                vol_ratio_val < 0.8):            # Low volume (losing momentum)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: trend reversal or momentum fade
            if (close_val > ema_val or           # Price above 4h EMA50
                rsi_val < 25 or                  # 1d RSI oversold
                vol_ratio_val < 0.8):            # Low volume (losing momentum)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals