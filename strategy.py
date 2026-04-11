#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_breakout_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return signals
    
    # Calculate weekly KAMA for trend filter
    close_1w = df_1w['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1w, k=10))
    volatility = np.sum(np.abs(np.diff(close_1w, k=1)), axis=0)
    # Pad arrays for alignment
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(1, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close_1w, np.nan)
    kama[9] = close_1w[9]  # Start after 10 periods
    for i in range(10, len(close_1w)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Calculate weekly RSI(14)
    delta = np.diff(close_1w)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Pad first value
    gain = np.concatenate([[np.nan], gain])
    loss = np.concatenate([[np.nan], loss])
    # Wilder's smoothing
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    avg_gain[14] = np.nanmean(gain[1:15])
    avg_loss[14] = np.nanmean(loss[1:15])
    for i in range(15, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily volume confirmation: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly indicators to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 1.5 * vol_ma_20[i]
        
        # KAMA trend filter: price above/below KAMA
        price_above_kama = price_close > kama_aligned[i]
        price_below_kama = price_close < kama_aligned[i]
        
        # RSI conditions
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        
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

# Hypothesis: 1d KAMA+RSI strategy with weekly trend filter for BTC/ETH in both bull and bear markets.
# Uses weekly KAMA to determine primary trend (price above/below) and weekly RSI for overbought/oversold signals.
# Enters long when price is above weekly KAMA, weekly RSI < 30 (oversold), and daily volume > 1.5x 20-day average.
# Enters short when price is below weekly KAMA, weekly RSI > 70 (overbought), and daily volume > 1.5x 20-day average.
# Exits when RSI returns to neutral (50) to avoid whipsaws in ranging markets.
# Weekly timeframe provides strong trend filter to reduce false signals, while daily entries capture timely entries.
# Volume confirmation ensures institutional participation. Position size 0.25 manages risk.
# Target: 10-20 trades per year (40-80 total over 4 years) to minimize fee drag.
# Works in bull markets (trend following via KAMA) and bear markets (mean reversion via RSI extremes).