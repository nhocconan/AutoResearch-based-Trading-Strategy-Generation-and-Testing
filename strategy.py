#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    # Hypothesis: Weekly high/low breakouts on daily chart with volume confirmation and RSI filter.
    # The weekly high and low act as significant support/resistance levels.
    # A breakout above weekly high with increasing volume suggests bullish momentum,
    # while a breakout below weekly low suggests bearish momentum.
    # RSI filter avoids entering at extreme overbought/oversold levels.
    # This strategy aims to capture medium-term trends while keeping trade frequency low
    # to minimize fee drag, suitable for both bull and bear markets.
    
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once before loop
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values  # not used but kept for clarity
    
    # Align weekly high/low to daily timeframe (values available after weekly bar closes)
    high_weekly_aligned = align_htf_to_ltf(prices, df_weekly, high_weekly)
    low_weekly_aligned = align_htf_to_ltf(prices, df_weekly, low_weekly)
    
    # Daily RSI(14)
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)  # avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    # Daily average volume (20-period)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after warmup for volume MA and RSI
        # Skip if weekly data not ready
        if np.isnan(high_weekly_aligned[i]) or np.isnan(low_weekly_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        rsi_val = rsi[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_filter = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: breakout above weekly high, not overbought, volume confirmation
            if price > high_weekly_aligned[i] and rsi_val < 60 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: breakout below weekly low, not oversold, volume confirmation
            elif price < low_weekly_aligned[i] and rsi_val > 40 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            if position == 1:  # long
                # Exit when price breaks below weekly low
                if price < low_weekly_aligned[i]:
                    exit_signal = True
            elif position == -1:  # short
                # Exit when price breaks above weekly high
                if price > high_weekly_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyHighLow_Breakout_VolumeRSI"
timeframe = "1d"
leverage = 1.0