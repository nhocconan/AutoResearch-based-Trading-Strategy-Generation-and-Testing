#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot with 1d trend filter and volume confirmation.
# Enters long when price touches Camarilla L3 support with bullish 1d trend and expanding volume.
# Enters short when price touches Camarilla H3 resistance with bearish 1d trend and expanding volume.
# Uses ATR(14) for dynamic stoploss and position sizing.
# Designed for 12-37 trades/year on 12h timeframe with focus on mean reversion at key levels.
# Volume filter ensures institutional participation, reducing false signals.
# 1d trend filter prevents counter-trend trading in choppy markets.

name = "12h_1d_camarilla_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for volatility filtering and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume moving average (10-period)
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    # Calculate Camarilla levels from previous 1d period
    # Need previous day's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # For first bar, use first available values
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla formulas
    # H3 = C + (H-L)*1.1/4
    # L3 = C - (H-L)*1.1/4
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(10, n):  # Start after EMA period
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14[i]) or
            np.isnan(vol_ma_10[i]) or np.isnan(camarilla_h3[i]) or
            np.isnan(camarilla_l3[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.2 * 10-period average volume
        vol_filter = volume[i] > 1.2 * vol_ma_10[i]
        
        # Determine 1d trend direction
        is_bullish_trend = close[i] > ema_50_1d_aligned[i]
        is_bearish_trend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions: price touches Camarilla levels with volume and trend
        # Long when price touches L3 support with bullish trend
        long_entry = (low[i] <= camarilla_l3[i] * 1.001) and vol_filter and is_bullish_trend
        # Short when price touches H3 resistance with bearish trend
        short_entry = (high[i] >= camarilla_h3[i] * 0.999) and vol_filter and is_bearish_trend
        
        # Exit conditions: opposite signal or trend reversal
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long on bearish signal or trend reversal
            exit_long = short_entry or not is_bullish_trend
        elif position == -1:
            # Exit short on bullish signal or trend reversal
            exit_short = long_entry or not is_bearish_trend
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals