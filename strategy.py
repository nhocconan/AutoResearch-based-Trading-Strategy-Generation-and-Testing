#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h RSI(2) mean reversion with 1d trend filter and volume confirmation
# Long when RSI(2) < 10, price > 1d EMA200, and volume > 1.5x 20-day average
# Short when RSI(2) > 90, price < 1d EMA200, and volume > 1.5x 20-day average
# Exit when RSI(2) crosses back to neutral (40-60 range)
# RSI(2) captures extreme short-term reversals
# 1d EMA200 ensures we trade with the higher timeframe trend
# Volume confirmation ensures conviction behind the move
# Target: 50-150 total trades over 4 years (12-37/year) by requiring multiple confluence factors

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 6-period RSI(2) equivalent using close prices
    # RSI(2) = 100 - (100 / (1 + RS)), where RS = avg_gain / avg_loss over 2 periods
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (alpha = 1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi2 = 100 - (100 / (1 + rs))
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 6h timeframe
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(2, n):  # Start after RSI(2) warmup
        # Skip if data not ready
        if (np.isnan(rsi2[i]) or np.isnan(ema200_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        rsi = rsi2[i]
        price = close[i]
        ema200_val = ema200_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        
        # Get current 1d volume (6 bars per day)
        day_idx = i // 6
        if day_idx < len(df_1d):
            volume = df_1d['volume'].iloc[day_idx]
        else:
            volume = df_1d['volume'].iloc[-1] if len(df_1d) > 0 else 0
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        volume_confirm = volume > 1.5 * vol_ma if vol_ma > 0 else False
        
        if position == 0:
            # Long: RSI(2) < 10 (oversold), price > EMA200 (uptrend), volume confirmation
            if rsi < 10 and price > ema200_val and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: RSI(2) > 90 (overbought), price < EMA200 (downtrend), volume confirmation
            elif rsi > 90 and price < ema200_val and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: RSI returns to neutral zone (40-60)
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if RSI crosses back above 40 (leaving oversold territory)
                if rsi >= 40:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if RSI crosses back below 60 (leaving overbought territory)
                if rsi <= 60:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_RSI2_MeanReversion_1dEMA200_Trend_Volume"
timeframe = "6h"
leverage = 1.0