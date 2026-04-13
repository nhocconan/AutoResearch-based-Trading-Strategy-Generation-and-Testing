#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d breakout above weekly Keltner upper band with weekly trend filter and volume confirmation.
# Long: price > weekly Keltner upper band + weekly EMA trend up + volume > 1.5x avg volume
# Short: price < weekly Keltner lower band + weekly EMA trend down + volume > 1.5x avg volume
# Weekly Keltner: EMA(20) +/- 2*ATR(10) on weekly data
# Weekly trend: EMA(50) slope (rising/falling)
# Volume confirmation reduces false breakouts
# Target: 20-60 total trades over 4 years (5-15/year) for 1d timeframe
# Works in bull/bear markets via EMA trend filter and volatility-based bands

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for Keltner and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA(20) for Keltner center
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Weekly ATR(10) for Keltner width
    tr_1w = np.maximum(
        high_1w[1:] - low_1w[1:],
        np.maximum(
            np.abs(high_1w[1:] - close_1w[:-1]),
            np.abs(low_1w[1:] - close_1w[:-1])
        )
    )
    tr_1w = np.concatenate([[np.nan], tr_1w])
    atr_10 = pd.Series(tr_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Weekly Keltner bands
    keltner_upper = ema_20 + 2.0 * atr_10
    keltner_lower = ema_20 - 2.0 * atr_10
    
    # Weekly EMA(50) for trend filter
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Trend: rising if current > previous, falling if current < previous
    ema_50_rising = np.zeros_like(ema_50, dtype=bool)
    ema_50_falling = np.zeros_like(ema_50, dtype=bool)
    ema_50_rising[1:] = ema_50[1:] > ema_50[:-1]
    ema_50_falling[1:] = ema_50[1:] < ema_50[:-1]
    
    # Align weekly indicators to daily
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1w, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1w, keltner_lower)
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_50_rising.astype(float))
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_50_falling.astype(float))
    
    # Daily average volume (20-day) for volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or
            np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        keltner_up = keltner_upper_aligned[i]
        keltner_low = keltner_lower_aligned[i]
        ema50_up = ema_50_rising_aligned[i] > 0.5
        ema50_down = ema_50_falling_aligned[i] > 0.5
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: break above weekly Keltner upper + weekly uptrend + volume confirmation
            if (price > keltner_up and 
                ema50_up and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: break below weekly Keltner lower + weekly downtrend + volume confirmation
            elif (price < keltner_low and 
                  ema50_down and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below weekly Keltner lower or weekly trend turns down
            if (price < keltner_low or
                ema50_down):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above weekly Keltner upper or weekly trend turns up
            if (price > keltner_up or
                ema50_up):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Keltner_Trend_Volume"
timeframe = "1d"
leverage = 1.0