#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h volume filter and 1d trend filter.
# In both bull and bear markets, price tends to revert to mean during consolidation.
# RSI < 30 (oversold) for long, RSI > 70 (overbought) for short.
# Volume filter ensures we trade on institutional interest.
# 1d EMA50 trend filter prevents trading against major trend.
# Timeframe: 1h for entries, 4h for volume confirmation, 1d for trend direction.
# Target: 20-30 trades/year per symbol to minimize fee drag.
name = "1h_RSI14_4hVolume_1dEMA50_MeanReversion"
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
    
    # Get 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 14-period RSI
    def calculate_rsi(close_prices, period=14):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)
        
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(close_prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Calculate EMA50 on daily
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 4h volume > 1.8x 20-period average
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 1h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure EMA50 and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        ema_50_val = ema_50_aligned[i]
        vol_ma = vol_ma_20_aligned[i]
        vol_4h = volume_4h[i // 4] if i >= 4 else 0  # 4 bars per 1h in 4h
        
        # Volume confirmation: current 4h volume > 1.8x 20-period average
        volume_confirmed = vol_4h > 1.8 * vol_ma
        
        if position == 0:
            # Enter long when RSI < 30 (oversold), price above EMA50 (uptrend), and volume confirmation
            if rsi_val < 30 and price > ema_50_val and volume_confirmed:
                signals[i] = 0.20
                position = 1
            # Enter short when RSI > 70 (overbought), price below EMA50 (downtrend), and volume confirmation
            elif rsi_val > 70 and price < ema_50_val and volume_confirmed:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long when RSI > 50 (mean reversion complete) or price crosses below EMA50
            if rsi_val > 50 or price < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short when RSI < 50 (mean reversion complete) or price crosses above EMA50
            if rsi_val < 50 or price > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals