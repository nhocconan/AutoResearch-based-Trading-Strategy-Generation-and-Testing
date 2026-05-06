#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h RSI for momentum and 1d ATR for volatility filtering
# - Uses 4h RSI(14) to identify momentum extremes (oversold <30, overbought >70)
# - Uses 1d ATR(14) normalized by price to filter for adequate volatility
# - Enters long when RSI crosses above 30 from below with adequate volatility
# - Enters short when RSI crosses below 70 from above with adequate volatility
# - Exits when RSI returns to neutral zone (40-60)
# - Designed to capture mean reversion moves during volatile periods with momentum confirmation
# - Target: 60-150 total trades over 4 years (15-37/year) with 0.20 position sizing
# - Uses 4h for signal direction (RSI momentum), 1h only for entry timing
# - Session filter: 08-20 UTC to reduce noise

name = "1h_4hRSI_1dATR_MeanReversion"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for RSI calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Get 1d data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 4h RSI (14)
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    gain_smoothed = wilders_smoothing(gain, 14)
    loss_smoothed = wilders_smoothing(loss, 14)
    rs = gain_smoothed / (loss_smoothed + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    
    # Calculate 1d ATR (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr_14 = wilders_smoothing(tr, 14)
    # Normalize ATR by price to get volatility percentage
    atr_percent = atr_14 / close_1d
    
    # Align 4h RSI to 1h timeframe
    rsi_4h_1h = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Align 1d ATR percent to 1h timeframe
    atr_percent_1h = align_htf_to_ltf(prices, df_1d, atr_percent)
    
    # Calculate RSI cross signals
    rsi_above_30 = rsi_4h_1h > 30
    rsi_below_70 = rsi_4h_1h < 70
    rsi_cross_above_30 = (rsi_above_30 & ~np.roll(rsi_above_30, 1)) | ((np.arange(len(rsi_above_30)) == 0) & rsi_above_30[0])
    rsi_cross_below_70 = (~rsi_below_70 & np.roll(rsi_below_70, 1)) | ((np.arange(len(rsi_below_70)) == 0) & ~rsi_below_70[0])
    
    # Volatility filter: ATR percent > 1% (adjustable threshold)
    vol_filter = atr_percent_1h > 0.01
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(rsi_4h_1h[i]) or np.isnan(atr_percent_1h[i]) or 
            np.isnan(rsi_cross_above_30[i]) or np.isnan(rsi_cross_below_70[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for RSI cross with volatility and session filters
            if rsi_cross_above_30[i] and vol_filter[i] and session_filter[i]:
                signals[i] = 0.20
                position = 1
            elif rsi_cross_below_70[i] and vol_filter[i] and session_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI returns to neutral zone (above 40)
            if rsi_4h_1h[i] >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI returns to neutral zone (below 60)
            if rsi_4h_1h[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals