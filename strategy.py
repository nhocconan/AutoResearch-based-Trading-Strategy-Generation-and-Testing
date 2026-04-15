#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Keltner Channel Breakout with Volume and Momentum Confirmation
# Uses 1w Keltner Channel (EMA10 + ATR*2) for trend context. Long when price breaks above upper band
# with volume > 1.5x average and RSI > 50 (bullish momentum). Short when price breaks below lower band
# with volume > 1.5x average and RSI < 50 (bearish momentum). Designed for low trade frequency
# and robust performance in both bull and bear markets by requiring strong momentum confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for Keltner Channel
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate EMA(10) on weekly close
    ema_10 = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate ATR(10) on weekly data
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Channel: EMA(10) ± ATR*2
    upper_kc = ema_10 + 2 * atr
    lower_kc = ema_10 - 2 * atr
    
    # Align Keltner Channel to daily timeframe
    upper_kc_aligned = align_htf_to_ltf(prices, df_1w, upper_kc)
    lower_kc_aligned = align_htf_to_ltf(prices, df_1w, lower_kc)
    
    # Calculate daily RSI(14) for momentum confirmation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_kc_aligned[i]) or np.isnan(lower_kc_aligned[i]) or
            np.isnan(rsi[i])):
            continue
        
        # Long entry: price breaks above upper Keltner + volume confirmation + bullish momentum
        if (close[i] > upper_kc_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            rsi[i] > 50 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below lower Keltner + volume confirmation + bearish momentum
        elif (close[i] < lower_kc_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              rsi[i] < 50 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite Keltner band touch or RSI crosses 50 (momentum shift)
        elif position == 1 and (close[i] < lower_kc_aligned[i] or rsi[i] < 50):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > upper_kc_aligned[i] or rsi[i] > 50):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_Weekly_Keltner_Breakout_Volume_Momentum"
timeframe = "1d"
leverage = 1.0