#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams Alligator + 1d RSI mean reversion filter
    # Uses Alligator jaws/teeth/lips to identify trend, enters on pullbacks in trend direction
    # RSI filter avoids overextended entries. Works in bull/bear via trend alignment.
    # Target: 15-30 trades/year per symbol to minimize fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 12h data for Williams Alligator (SMMA-based)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Williams Alligator: three SMMA lines (Jaw=13, Teeth=8, Lips=5)
    def smma(series, period):
        sma = np.full_like(series, np.nan, dtype=float)
        if len(series) < period:
            return sma
        sma[period-1] = np.mean(series[:period])
        for i in range(period, len(series)):
            sma[i] = (sma[i-1] * (period-1) + series[i]) / period
        return sma
    
    jaw = smma(close_12h, 13)
    teeth = smma(close_12h, 8)
    lips = smma(close_12h, 5)
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Load 1d data for RSI(14)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # RSI calculation
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(gain, np.nan, dtype=float)
    avg_loss = np.full_like(loss, np.nan, dtype=float)
    
    # Wilder's smoothing
    avg_gain[13] = np.mean(gain[1:14])  # first 14 periods
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(rsi_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaw (uptrend)
            bullish = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
            # Bearish alignment: Jaw > Teeth > Lips (downtrend)
            bearish = jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]
            
            # Long: Pullback in uptrend when RSI < 40 (not overbought)
            if bullish and rsi_aligned[i] < 40:
                signals[i] = 0.25
                position = 1
            # Short: Pullback in downtrend when RSI > 60 (not oversold)
            elif bearish and rsi_aligned[i] > 60:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Trend reversal or RSI extreme
            if position == 1:
                bearish = jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]
                if bearish or rsi_aligned[i] > 70:  # trend change or overbought
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                bullish = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
                if bullish or rsi_aligned[i] < 30:  # trend change or oversold
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_RSI_Pullback_v1"
timeframe = "12h"
leverage = 1.0