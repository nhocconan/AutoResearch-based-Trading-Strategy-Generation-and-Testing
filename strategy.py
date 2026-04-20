#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_HullEMA_RSI_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 21 or len(df_1d) < 14:
        return np.zeros(n)
    
    # === 4h: Hull Moving Average (HMA21) for trend ===
    close_4h = df_4h['close'].values
    # HMA(n) = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_n = 21 // 2
    sqrt_n = int(np.sqrt(21))
    wma_half = pd.Series(close_4h).ewm(span=half_n, adjust=False, min_periods=half_n).mean().values
    wma_full = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_21 = pd.Series(raw_hma).ewm(span=sqrt_n, adjust=False, min_periods=sqrt_n).mean().values
    hma_21_aligned = align_htf_to_ltf(prices, df_4h, hma_21)
    
    # === 1d: RSI(14) for momentum ===
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rs = np.where(avg_loss == 0, 100, rs)  # Avoid division by zero
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 1h: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
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
        hma_val = hma_21_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is invalid
        if np.isnan(hma_val) or np.isnan(rsi_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Uptrend + bullish momentum + volume confirmation
            if (close_val > hma_val and          # Price above 4h HMA21 (uptrend)
                50 < rsi_val < 70 and            # 1d RSI in bullish momentum
                vol_ratio_val > 1.5):            # Volume confirmation
                signals[i] = 0.20
                position = 1
            # Short: Downtrend + bearish momentum + volume confirmation
            elif (close_val < hma_val and        # Price below 4h HMA21 (downtrend)
                  30 < rsi_val < 50 and          # 1d RSI in bearish momentum
                  vol_ratio_val > 1.5):          # Volume confirmation
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: trend breakdown or momentum fade
            if (close_val < hma_val or           # Price below 4h HMA21
                rsi_val > 75 or                  # 1d RSI overbought
                vol_ratio_val < 0.8):            # Low volume (losing momentum)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: trend reversal or momentum fade
            if (close_val > hma_val or           # Price above 4h HMA21
                rsi_val < 25 or                  # 1d RSI oversold
                vol_ratio_val < 0.8):            # Low volume (losing momentum)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals