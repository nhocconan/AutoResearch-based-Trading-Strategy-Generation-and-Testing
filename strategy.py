# 1d_RSI20_CamPivot_Breakout_1wTrend
# Hypothesis: Daily RSI(20) extremes + weekly trend filter + Camarilla pivot breakout with volume confirmation
# Works in bull (breaks above R3 in uptrend) and bear (breaks below S3 in downtrend)
# Weekly trend avoids whipsaws, RSI prevents overextension, volume confirms breakout strength
# Target: 15-25 trades/year, low frequency to minimize fee drag on 1d timeframe

name = "1d_RSI20_CamPivot_Breakout_1wTrend"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Get daily data for Camarilla pivots (from previous daily bar)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range
    range_1d = high_1d - low_1d
    
    # Calculate Camarilla R3 and S3 levels (stronger breakout levels)
    camarilla_r3 = close_1d + (range_1d * 1.1 / 4)
    camarilla_s3 = close_1d - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to daily timeframe (using previous day's values)
    r3_1d = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_1d = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate daily RSI(20)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/20, min_periods=20).mean()
    avg_loss = loss.ewm(alpha=1/20, min_periods=20).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_1d[i]) or np.isnan(s3_1d[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) AND price breaks above R3 AND weekly uptrend AND volume surge
            if (rsi_values[i] < 30 and close[i] > r3_1d[i] and 
                close[i] > ema_1w_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) AND price breaks below S3 AND weekly downtrend AND volume surge
            elif (rsi_values[i] > 70 and close[i] < s3_1d[i] and 
                  close[i] < ema_1w_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI > 70 (overbought) OR price falls below S3 OR weekly trend turns down
            if (rsi_values[i] > 70 or close[i] < s3_1d[i] or 
                close[i] < ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: RSI < 30 (oversold) OR price rises above R3 OR weekly trend turns up
            if (rsi_values[i] < 30 or close[i] > r3_1d[i] or 
                close[i] > ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals