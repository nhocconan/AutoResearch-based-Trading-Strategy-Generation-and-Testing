#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_RSI_Trend_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily RSI(14) for trend filter ===
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    delta = close_1d_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_1d = rsi_14.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 6h: Price and EMA(34) ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    close_series = pd.Series(close)
    ema34 = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume ratio (current vs 30-period average)
    vol_series = pd.Series(volume)
    vol_ma30 = vol_series.rolling(window=30, min_periods=30).mean().values
    vol_ratio = volume / np.where(vol_ma30 > 0, vol_ma30, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = close[i]
        ema34_val = ema34[i]
        rsi_1d_val = rsi_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(ema34_val) or np.isnan(rsi_1d_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Daily RSI > 50 (bullish bias) and price > EMA34 with volume confirmation
            if rsi_1d_val > 50 and close_val > ema34_val and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Daily RSI < 50 (bearish bias) and price < EMA34 with volume confirmation
            elif rsi_1d_val < 50 and close_val < ema34_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price falls below EMA34 or daily RSI turns bearish (< 40)
            if close_val < ema34_val or rsi_1d_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises above EMA34 or daily RSI turns bullish (> 60)
            if close_val > ema34_val or rsi_1d_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals