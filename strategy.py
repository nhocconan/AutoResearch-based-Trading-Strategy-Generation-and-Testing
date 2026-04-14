#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams Alligator for trend direction and 1w RSI for momentum confirmation.
# Long when price is above Alligator teeth (SMA13) with 1w RSI > 50 (uptrend momentum).
# Short when price is below Alligator teeth with 1w RSI < 50 (downtrend momentum).
# Exit when price crosses back below/above Alligator teeth or RSI crosses 50 in opposite direction.
# Williams Alligator consists of three SMAs: Jaw (13-period, 8-bar shift), Teeth (8-period, 5-bar shift), Lips (5-period, 3-bar shift).
# Uses close price for simplicity; Teeth line (SMA8 shifted 5) acts as dynamic support/resistance.
# Designed to capture trends in both bull and bear markets by aligning price with smoothed momentum.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Load 1d data ONCE for Williams Alligator (Teeth line: SMA8 shifted 5)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Williams Alligator Teeth: SMA(8) shifted by 5 periods
    # Teeth = SMA(Median Price, 8) shifted 5 bars forward
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    sma8_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values
    teeth_1d = np.roll(sma8_1d, 5)  # shift forward by 5 bars
    teeth_1d[:5] = np.nan  # first 5 values invalid due to shift
    
    # Load 1w data ONCE for RSI momentum filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # RSI(14) on 1w
    delta = np.diff(close_1w, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, 100, rs)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Align indicators to lower timeframe (12h)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(13, 14)  # Need Alligator and RSI
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(rsi_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above Alligator teeth AND 1w RSI > 50 (uptrend momentum)
            if (close[i] > teeth_1d_aligned[i] and 
                rsi_1w_aligned[i] > 50):
                position = 1
                signals[i] = position_size
            # Short: price below Alligator teeth AND 1w RSI < 50 (downtrend momentum)
            elif (close[i] < teeth_1d_aligned[i] and 
                  rsi_1w_aligned[i] < 50):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Alligator teeth OR RSI crosses below 50
            if (close[i] <= teeth_1d_aligned[i] or 
                rsi_1w_aligned[i] <= 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Alligator teeth OR RSI crosses above 50
            if (close[i] >= teeth_1d_aligned[i] or 
                rsi_1w_aligned[i] >= 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_WilliamsAlligator_1wRSI_v1"
timeframe = "12h"
leverage = 1.0