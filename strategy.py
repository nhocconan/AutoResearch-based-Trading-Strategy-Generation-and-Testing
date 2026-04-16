# 4h_RSI_Trend_Confirmation
# Hypothesis: RSI(14) combined with 4h EMA(50) trend filter provides reliable entries in both bull and bear markets.
# Long: RSI crosses above 50 + price above EMA(50) + volume confirmation.
# Short: RSI crosses below 50 + price below EMA(50) + volume confirmation.
# Uses 4h timeframe for execution, 1d for volume confirmation and trend confirmation.
# Target: 100-150 trades over 4 years (25-38/year) with disciplined entries.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (primary) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (HTF for volume confirmation) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # === RSI(14) on 4h close ===
    delta = np.diff(close_4h)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_4h, np.nan)
    avg_loss = np.full_like(close_4h, np.nan)
    
    # Wilder's smoothing for RSI
    period = 14
    if len(gain) >= period:
        avg_gain[period-1] = np.mean(gain[:period])
        avg_loss[period-1] = np.mean(loss[:period])
        for i in range(period, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === EMA(50) on 4h close ===
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === Volume confirmation: 1d volume ratio ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_20_1d
    
    # Align all HTF data to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    
    # Warmup: enough for RSI and EMA calculations
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        rsi_val = rsi_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_ratio = vol_ratio_1d_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: RSI crosses below 50 OR price closes below EMA(50)
            if rsi_val < 50 or price < ema_val:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: RSI crosses above 50 OR price closes above EMA(50)
            if rsi_val > 50 or price > ema_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: RSI crosses above 50 + price above EMA(50) + volume confirmation
            if rsi_val > 50 and price > ema_val and vol_ratio > 1.2:
                signals[i] = 0.25
                position = 1
                continue
            # SHORT: RSI crosses below 50 + price below EMA(50) + volume confirmation
            elif rsi_val < 50 and price < ema_val and vol_ratio > 1.2:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_RSI_Trend_Confirmation"
timeframe = "4h"
leverage = 1.0