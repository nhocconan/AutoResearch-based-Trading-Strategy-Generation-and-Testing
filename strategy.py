#!/usr/bin/env python3
"""
Hypothesis: 1h RSI mean reversion with 4h/1d trend filter and volume confirmation.
Long when RSI(14) < 30 AND price > 4h EMA(50) AND price > 1d EMA(200) AND volume > 1.5x 20-period average.
Short when RSI(14) > 70 AND price < 4h EMA(50) AND price < 1d EMA(200) AND volume > 1.5x 20-period average.
Exit when RSI crosses 50 (mean reversion complete) or volume drops below average.
Uses proven RSI mean reversion with multi-timeframe trend alignment.
Designed for low trade frequency (15-37/year) on 1h timeframe to minimize fee drag.
Session filter: 08-20 UTC to reduce noise trades.
Position size: 0.20 (discrete level to minimize churn).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Get 1d data for long-term trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Use Wilder's smoothing (alpha = 1/period)
    alpha = 1.0 / 14
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First avg gain
    avg_loss[13] = np.mean(loss[1:14])  # First avg loss
    
    for i in range(14, len(gain)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:13] = np.nan  # Not enough data for first 13 periods
    
    # Calculate 4h EMA(50)
    ema_4h_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    # Calculate 1d EMA(200)
    ema_1d_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_200)
    
    # Calculate volume average (20-period) on 1h
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_50_aligned[i]) or np.isnan(ema_1d_200_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        ema_4h = ema_4h_50_aligned[i]
        ema_1d = ema_1d_200_aligned[i]
        vol_ma = volume_ma[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: RSI oversold AND price above both EMAs AND volume > 1.5x avg
            if rsi_val < 30 and price > ema_4h and price > ema_1d and vol > 1.5 * vol_ma:
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought AND price below both EMAs AND volume > 1.5x avg
            elif rsi_val > 70 and price < ema_4h and price < ema_1d and vol > 1.5 * vol_ma:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI crosses above 50 (mean reversion) OR volume drops below average
            if rsi_val > 50 or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI crosses below 50 (mean reversion) OR volume drops below average
            if rsi_val < 50 or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI_MeanReversion_4h1dTrend_Volume"
timeframe = "1h"
leverage = 1.0