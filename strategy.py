#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_breakout_v2"
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
    change = np.abs(np.diff(close_1w, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1w, n=1)), axis=0)  # 10-period sum of abs changes
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.divide(change, volatility, out=np.full_like(change, 0.1), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.full_like(close_1w, np.nan)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Calculate RSI on weekly data
    delta = np.diff(close_1w)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Wilder's smoothing
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    avg_gain[14] = np.nanmean(gain[1:15])  # first 14 avg
    avg_loss[14] = np.nanmean(loss[1:15])
    for i in range(15, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 0), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align weekly indicators to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    # Daily volume confirmation: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 1.5 * vol_ma_20[i]
        
        # Trend direction from KAMA
        trend_up = price_close > kama_aligned[i]
        trend_down = price_close < kama_aligned[i]
        
        # RSI conditions
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: KAMA uptrend + RSI oversold + volume confirmation
        if trend_up and rsi_oversold and vol_confirm:
            enter_long = True
        
        # Short: KAMA downtrend + RSI overbought + volume confirmation
        if trend_down and rsi_overbought and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite RSI extreme or trend reversal
        exit_long = (rsi_aligned[i] > 70) or (price_close < kama_aligned[i])
        exit_short = (rsi_aligned[i] < 30) or (price_close > kama_aligned[i])
        
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

# Hypothesis: 1d KAMA + RSI strategy with weekly trend filter for BTC/ETH.
# Uses weekly KAMA for trend direction and weekly RSI for overbought/oversold conditions.
# Enters long when: price > weekly KAMA (uptrend) + weekly RSI < 30 (oversold) + volume > 1.5x 20-day average.
# Enters short when: price < weekly KAMA (downtrend) + weekly RSI > 70 (overbought) + volume > 1.5x 20-day average.
# Exits when RSI reaches opposite extreme or price crosses weekly KAMA.
# Weekly timeframe filter reduces whipsaws and aligns with major trends.
# Volume confirmation ensures participation during significant moves.
# Position size 0.25 manages risk while allowing meaningful returns.
# Target: 15-25 trades per year (60-100 total over 4 years) to minimize fee drag.
# Works in bull markets (trend following) and bear markets (mean reversion at extremes).