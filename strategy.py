#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 14-day ATR on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR uses only high-low
    
    atr_14 = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if i < 14:
            atr_14[i] = np.nan
        elif i == 14:
            atr_14[i] = np.mean(tr[1:15])
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate Bollinger Bands on daily close (20, 2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    bb_width = upper_band - lower_band
    
    # Calculate Bollinger Band Width percentile over 50 days to identify squeeze
    bb_width_percentile = np.full_like(bb_width, np.nan)
    for i in range(50, len(bb_width)):
        window = bb_width[i-50:i]
        if not np.all(np.isnan(window)):
            valid_window = window[~np.isnan(window)]
            if len(valid_window) > 0:
                bb_width_percentile[i] = (np.sum(valid_window <= bb_width[i]) / len(valid_window)) * 100
    
    # Align BB width percentile to 4h timeframe
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    # Calculate RSI on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    for i in range(len(gain)):
        if i < 14:
            avg_gain[i] = np.nan
            avg_loss[i] = np.nan
        elif i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 14)  # 50 for BB width percentile, 14 for RSI
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bb_width_percentile_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bb_width_percentile_val = bb_width_percentile_aligned[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Bollinger Band Squeeze breakout with RSI confirmation
            # Long: BB width in lower 20% (squeeze) + RSI > 50 (bullish momentum)
            if bb_width_percentile_val < 20 and rsi_val > 50:
                position = 1
                signals[i] = position_size
            # Short: BB width in lower 20% (squeeze) + RSI < 50 (bearish momentum)
            elif bb_width_percentile_val < 20 and rsi_val < 50:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: BB width expands above 80% OR RSI becomes overbought (>70)
            if bb_width_percentile_val > 80 or rsi_val > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: BB width expands above 80% OR RSI becomes oversold (<30)
            if bb_width_percentile_val > 80 or rsi_val < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_BB_Width_Squeeze_RSI"
timeframe = "4h"
leverage = 1.0