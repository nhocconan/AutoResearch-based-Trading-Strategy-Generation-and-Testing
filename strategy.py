#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily KAMA direction with weekly RSI filter and volume confirmation.
# Long when KAMA(14) is rising (bullish) and weekly RSI(14) > 50 with volume > 1.5x average.
# Short when KAMA(14) is falling (bearish) and weekly RSI(14) < 50 with volume > 1.5x average.
# Uses weekly RSI to filter out counter-trend trades and avoid whipsaw in sideways markets.
# Volume confirmation ensures momentum has institutional participation.
# Target: 15-25 trades/year per symbol (~60-100 total over 4 years).
name = "1d_KAMA14_WeeklyRSI_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for RSI calculation
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly RSI(14)
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Align weekly RSI to daily timeframe (wait for weekly close)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate daily KAMA(14)
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close, k=14, prepend=close[:14]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])).reshape(-1, 14), axis=1)
    er = change / (volatility + 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20)  # Need KAMA and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(vol_ma_20[i]) or i < 14):  # KAMA needs at least 14 periods
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi_1w_aligned[i]
        kama_val = kama[i]
        kama_prev = kama[i-1]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # KAMA direction: rising if current > previous
        kama_rising = kama_val > kama_prev
        kama_falling = kama_val < kama_prev
        
        if position == 0:
            # Enter long: KAMA rising AND weekly RSI > 50 AND volume confirmed
            if kama_rising and rsi_val > 50 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA falling AND weekly RSI < 50 AND volume confirmed
            elif kama_falling and rsi_val < 50 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when KAMA falls or weekly RSI < 50
            if kama_falling or rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when KAMA rises or weekly RSI > 50
            if kama_rising or rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals