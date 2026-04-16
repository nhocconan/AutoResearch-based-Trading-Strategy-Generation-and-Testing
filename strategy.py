#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d RSI mean reversion with 1w trend filter and volume spike confirmation.
# Long when 1d RSI(14) < 30 (oversold) and price > 4h EMA(50) with volume > 2x 20-period average.
# Short when 1d RSI(14) > 70 (overbought) and price < 4h EMA(50) with volume > 2x 20-period average.
# Exit when 1d RSI returns to 50 (neutral) or opposite extreme.
# Uses discrete position size 0.25. 1d RSI provides mean reversion edge, 4h EMA provides trend alignment.
# Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once before loop for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === Daily Indicators: RSI(14) ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing (alpha = 1/14)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 0.0), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align daily RSI to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Get weekly data once before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === Weekly Indicators: EMA(50) for trend filter ===
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # 4h EMA(50) for entry timing
    ema_50_4h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume moving average (20-period) on 4h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        rsi_val = rsi_aligned[i]
        ema_50_w = ema_50_aligned[i]
        ema_50_4h_val = ema_50_4h[i]
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if RSI returns to 50 (neutral) or becomes overbought
            if rsi_val >= 50:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if RSI returns to 50 (neutral) or becomes oversold
            if rsi_val <= 50:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Trend filter: price must be above/below 4h EMA(50) for alignment
            trend_filter_long = price > ema_50_4h_val
            trend_filter_short = price < ema_50_4h_val
            
            # Volume filter: volume spike > 2x 20-period average
            vol_filter = vol > 2.0 * vol_ma
            
            # LONG: 1d RSI oversold (<30) with trend and volume confirmation
            if (rsi_val < 30) and trend_filter_long and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: 1d RSI overbought (>70) with trend and volume confirmation
            elif (rsi_val > 70) and trend_filter_short and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_1dRSI14_1wEMA50_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0