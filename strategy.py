#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Trend_Momentum_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Weekly: Trend filter (EMA20) ===
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # === Weekly: Momentum filter (RSI14) ===
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_loss / np.where(avg_gain > 0, avg_gain, np.nan)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # === Daily: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema20_1w_aligned[i]
        rsi_val = rsi_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_val) or np.isnan(rsi_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Uptrend + bullish momentum + volume confirmation
            if (close_val > ema_val and          # Price above weekly EMA20 (uptrend)
                40 < rsi_val < 70 and            # Weekly RSI in bullish range
                vol_ratio_val > 1.5):            # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Downtrend + bearish momentum + volume confirmation
            elif (close_val < ema_val and        # Price below weekly EMA20 (downtrend)
                  30 < rsi_val < 60 and          # Weekly RSI in bearish range
                  vol_ratio_val > 1.5):          # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend breakdown or momentum fade
            if (close_val < ema_val or           # Price below weekly EMA20
                rsi_val > 75 or                  # Weekly RSI overbought
                vol_ratio_val < 0.8):            # Low volume (losing momentum)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or momentum fade
            if (close_val > ema_val or           # Price above weekly EMA20
                rsi_val < 25 or                  # Weekly RSI oversold
                vol_ratio_val < 0.8):            # Low volume (losing momentum)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals