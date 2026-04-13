#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day RSI + 1-week Bollinger Bands mean reversion with volume confirmation.
# RSI < 30 and price below BB lower band indicates oversold condition for long.
# RSI > 70 and price above BB upper band indicates overbought condition for short.
# Weekly trend filter (price above/below weekly EMA20) prevents counter-trend trades.
# Volume spike confirms institutional participation.
# Target: 15-25 trades per year (60-100 total over 4 years) for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-week data for Bollinger Bands and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma_1w = np.full(len(close_1w), np.nan)
    std_1w = np.full(len(close_1w), np.nan)
    
    for i in range(bb_period - 1, len(close_1w)):
        sma_1w[i] = np.mean(close_1w[i - bb_period + 1:i + 1])
        std_1w[i] = np.std(close_1w[i - bb_period + 1:i + 1])
    
    bb_upper = sma_1w + bb_std * std_1w
    bb_lower = sma_1w - bb_std * std_1w
    
    # Weekly EMA20 for trend filter
    ema_period = 20
    ema_multiplier = 2 / (ema_period + 1)
    ema_1w = np.zeros(len(close_1w))
    ema_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema_1w[i] = (close_1w[i] - ema_1w[i-1]) * ema_multiplier + ema_1w[i-1]
    
    # Align weekly indicators to daily timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_1w, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1w, bb_lower)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Daily RSI (14)
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    
    for i in range(rsi_period + 1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(rsi[i]) or np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        rsi_val = rsi[i]
        bb_upper_val = bb_upper_aligned[i]
        bb_lower_val = bb_lower_aligned[i]
        ema_trend = ema_1w_aligned[i]
        avg_vol = avg_volume[i]
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirm = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long: RSI < 30, price below BB lower, above weekly EMA, volume confirmation
            if (rsi_val < 30 and 
                price < bb_lower_val and 
                price > ema_trend and 
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: RSI > 70, price above BB upper, below weekly EMA, volume confirmation
            elif (rsi_val > 70 and 
                  price > bb_upper_val and 
                  price < ema_trend and 
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI > 50 or price crosses above weekly EMA
            if (rsi_val > 50 or price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI < 50 or price crosses below weekly EMA
            if (rsi_val < 50 or price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_RSI_BB_MeanReversion_Volume"
timeframe = "1d"
leverage = 1.0