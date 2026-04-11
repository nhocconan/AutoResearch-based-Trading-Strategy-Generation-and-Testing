#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Calculate KAMA on weekly data
    close_1w = df_1w['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.abs(np.diff(close_1w))
    er = np.zeros_like(close_1w)
    er[1:] = change[1:] / (np.sum(volatility[np.arange(1, len(close_1w))[:, None] <= np.arange(1, len(close_1w))], axis=1) + 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Calculate RSI on weekly data
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: weekly volume > 1.5x 20-period average
    vol_1w = df_1w['volume'].values
    vol_ma_20 = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly indicators to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation - moderate threshold
        vol_confirm = volume_current > 1.5 * vol_ma_20_aligned[i]
        
        # KAMA direction: price above/below KAMA
        price_above_kama = price_close > kama_aligned[i]
        price_below_kama = price_close < kama_aligned[i]
        
        # RSI conditions
        rsi_overbought = rsi_aligned[i] > 70
        rsi_oversold = rsi_aligned[i] < 30
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price above KAMA + RSI oversold + volume confirmation
        if price_above_kama and rsi_oversold and vol_confirm:
            enter_long = True
        
        # Short: Price below KAMA + RSI overbought + volume confirmation
        if price_below_kama and rsi_overbought and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite RSI extreme
        exit_long = rsi_aligned[i] > 50  # Exit long when RSI returns to neutral
        exit_short = rsi_aligned[i] < 50  # Exit short when RSI returns to neutral
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Daily KAMA/RSI strategy with weekly trend filter.
# Uses weekly KAMA for trend direction and weekly RSI for overbought/oversold signals.
# Enters long when price is above weekly KAMA (uptrend) + weekly RSI oversold (<30) + volume confirmation.
# Enters short when price is below weekly KAMA (downtrend) + weekly RSI overbought (>70) + volume confirmation.
# Exits when RSI returns to neutral (50) to avoid whipsaws.
# Volume confirmation requires 1.5x 20-period weekly average to filter low-quality signals.
# Position size 0.25 to balance risk and reward, targeting ~10-20 trades per year.
# Works in both bull and bear markets by following the weekly trend while fading short-term RSI extremes.
# Weekly timeframe reduces noise and provides institutional context for daily entries.