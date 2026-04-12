# 1h_4h_1d_rsi_volume_trend
# Hypothesis: Use 4h RSI for trend direction (oversold/overbought), 1d volume for conviction, and 1h for precise entry timing.
# Works in bull/bear by fading extremes in ranging markets and following momentum in trends.
# Target: 20-40 trades/year via strict RSI thresholds and volume confirmation.
# Timeframe: 1h, leverage: 1.0

name = "1h_4h_1d_rsi_volume_trend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def rsi(close, period=14):
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for RSI trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # 4h RSI (14)
    rsi_4h = rsi(close_4h, 14)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # 1d volume MA (20) for conviction
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi_4h_aligned[i]
        vol_ratio = vol_ratio_1d_aligned[i]
        
        # Long: 4h RSI oversold (<30) + above-average 1d volume + price above 1h open (intraday strength)
        if (rsi_val < 30 and vol_ratio > 1.0 and close[i] > prices['open'].iloc[i] and position != 1):
            position = 1
            signals[i] = 0.20
        # Short: 4h RSI overbought (>70) + above-average 1d volume + price below 1h open (intraday weakness)
        elif (rsi_val > 70 and vol_ratio > 1.0 and close[i] < prices['open'].iloc[i] and position != -1):
            position = -1
            signals[i] = -0.20
        # Exit: RSI returns to neutral zone (40-60) or volume drops
        elif position == 1 and (rsi_val > 40 or vol_ratio < 0.8):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi_val < 60 or vol_ratio < 0.8):
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals