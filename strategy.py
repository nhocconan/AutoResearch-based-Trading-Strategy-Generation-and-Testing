#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_RSI_Momentum_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # === 1d: RSI(14) for momentum ===
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss > 0, avg_loss, np.nan)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 1w: SMA(20) for long-term trend ===
    close_1w = df_1w['close'].values
    sma20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    sma20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma20_1w)
    
    # === 1d: Price and volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        close_val = close[i]
        vol_ratio_val = vol_ratio[i]
        rsi_val = rsi_1d_aligned[i]
        sma_val = sma20_1w_aligned[i]
        
        # Skip if any value is NaN
        if np.isnan(rsi_val) or np.isnan(sma_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Uptrend + bullish momentum + volume confirmation
            if (close_val > sma_val and          # Price above 1w SMA20 (long-term uptrend)
                50 < rsi_val < 70 and            # 1d RSI in bullish momentum (not overbought)
                vol_ratio_val > 1.5):            # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Downtrend + bearish momentum + volume confirmation
            elif (close_val < sma_val and        # Price below 1w SMA20 (long-term downtrend)
                  30 < rsi_val < 50 and          # 1d RSI in bearish momentum (not oversold)
                  vol_ratio_val > 1.5):          # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend breakdown or momentum fade
            if (close_val < sma_val or           # Price below 1w SMA20
                rsi_val > 75 or                  # 1d RSI overbought
                vol_ratio_val < 0.8):            # Low volume (losing momentum)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or momentum fade
            if (close_val > sma_val or           # Price above 1w SMA20
                rsi_val < 25 or                  # 1d RSI oversold
                vol_ratio_val < 0.8):            # Low volume (losing momentum)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals