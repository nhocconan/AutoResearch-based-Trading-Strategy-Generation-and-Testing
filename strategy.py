#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_RSI_MeanReversion_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Weekly RSI(14) for trend context ===
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_loss / np.where(avg_gain > 0, avg_gain, np.nan)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # === Daily data ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily RSI(2) for mean reversion signals
    delta_d = np.diff(close, prepend=close[0])
    gain_d = np.where(delta_d > 0, delta_d, 0)
    loss_d = np.where(delta_d < 0, -delta_d, 0)
    avg_gain_d = pd.Series(gain_d).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss_d = pd.Series(loss_d).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs_d = avg_loss_d / np.where(avg_gain_d > 0, avg_gain_d, np.nan)
    rsi_2 = 100 - (100 / (1 + rs_d))
    
    # Volume ratio (current vs 20-day average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Get values
        close_val = close[i]
        rsi_2_val = rsi_2[i]
        rsi_1w_val = rsi_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(rsi_2_val) or np.isnan(rsi_1w_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Extreme weekly weakness + daily oversold + volume confirmation
            if (rsi_1w_val < 30 and          # Weekly RSI oversold (bearish extreme)
                rsi_2_val < 10 and           # Daily RSI(2) extremely oversold
                vol_ratio_val > 1.5):        # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Extreme weekly strength + daily overbought + volume confirmation
            elif (rsi_1w_val > 70 and        # Weekly RSI overbought (bullish extreme)
                  rsi_2_val > 90 and         # Daily RSI(2) extremely overbought
                  vol_ratio_val > 1.5):      # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: weekly recovery or mean reversion
            if (rsi_1w_val > 50 or           # Weekly RSI recovered to neutral
                rsi_2_val > 50):             # Daily RSI(2) recovered to neutral
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly decline or mean reversion
            if (rsi_1w_val < 50 or           # Weekly RSI declined to neutral
                rsi_2_val < 50):             # Daily RSI(2) recovered to neutral
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals