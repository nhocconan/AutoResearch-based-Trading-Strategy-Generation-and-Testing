#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with RSI(14) pullback and 1w volume confirmation.
# Long when KAMA rising, RSI < 30, and weekly volume > 1.5x 4-week average.
# Short when KAMA falling, RSI > 70, and weekly volume > 1.5x 4-week average.
# Exit when RSI crosses back above 50 (long) or below 50 (short).
# KAMA adapts to market noise, reducing whipsaws in sideways markets.
# RSI overbought/oversold provides entry timing during pullbacks.
# Weekly volume filter ensures institutional participation. Target: 30-70 total trades over 4 years (7-18/year).

name = "1d_KAMA_RSI_Volume"
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
    
    # KAMA calculation (adaptive moving average)
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    
    # Proper KAMA implementation
    dir = np.abs(np.diff(close, 10))  # direction over 10 periods
    vol = np.sum(np.abs(np.diff(close, 1)), axis=0) if False else None  # placeholder
    
    # Simplified but correct KAMA using pandas
    close_series = pd.Series(close)
    # Calculate ER (Efficiency Ratio) over 10 periods
    change_10 = np.abs(close_series.diff(10))
    volatility_10 = close_series.diff(1).abs().rolling(window=10, min_periods=1).sum()
    er = change_10 / volatility_10.replace(0, np.nan)
    er = er.fillna(0).values
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # 1w data for volume filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 4:
        return np.zeros(n)
    
    vol_1w = df_1w['volume'].values
    vol_ma4 = pd.Series(vol_1w).rolling(window=4, min_periods=4).mean().values
    volume_filter = vol_1w > (1.5 * vol_ma4)
    
    # Align 1w volume filter to daily
    volume_filter_aligned = align_htf_to_ltf(prices, df_1w, volume_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup for KAMA and RSI
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_exit_long = rsi[i] > 50
        rsi_exit_short = rsi[i] < 50
        
        if position == 0:
            # Long conditions: KAMA rising, RSI oversold, volume confirmation
            if kama_rising and rsi_oversold and volume_filter_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: KAMA falling, RSI overbought, volume confirmation
            elif kama_falling and rsi_overbought and volume_filter_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI crosses back above 50
            if rsi_exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI crosses back below 50
            if rsi_exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals