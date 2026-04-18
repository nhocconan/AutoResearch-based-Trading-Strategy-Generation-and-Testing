#!/usr/bin/env python3
"""
4h_RSI2_Regime_Breakout
4h strategy using 2-period RSI for mean reversion in ranging markets and breakout in trending markets.
- Long: RSI2 < 10 + price > 200 EMA + volume > 1.5x 20-period volume MA (mean reversion long in uptrend)
- Short: RSI2 > 90 + price < 200 EMA + volume > 1.5x 20-period volume MA (mean reversion short in downtrend)
- Exit: Opposite RSI2 extreme or trend reversal
Designed for ~20-40 trades/year per symbol (80-160 total over 4 years)
Works in bull markets (mean reversion longs) and bear markets (mean reversion shorts)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Daily volume average (20-period)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily data to 4h timeframe
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 2-period RSI on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing: first average is simple average, then smoothed
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[1] = gain[1]  # first value after seed
    avg_loss[1] = loss[1]
    
    for i in range(2, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi2 = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # need enough for daily EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(vol_ma_aligned[i]) or
            np.isnan(rsi2[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions from daily timeframe
        uptrend = ema_200_aligned[i] > close[i]  # price above daily EMA200 = uptrend
        downtrend = ema_200_aligned[i] < close[i]  # price below daily EMA200 = downtrend
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # RSI2 extremes for mean reversion
        rsi_oversold = rsi2[i] < 10
        rsi_overbought = rsi2[i] > 90
        
        if position == 0:
            # Long: uptrend + volume + RSI2 oversold (mean reversion long)
            if uptrend and vol_confirm and rsi_oversold:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + RSI2 overbought (mean reversion short)
            elif downtrend and vol_confirm and rsi_overbought:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI2 overbought or trend reversal to downtrend
            if rsi_overbought or downtrend:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI2 oversold or trend reversal to uptrend
            if rsi_oversold or uptrend:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI2_Regime_Breakout"
timeframe = "4h"
leverage = 1.0