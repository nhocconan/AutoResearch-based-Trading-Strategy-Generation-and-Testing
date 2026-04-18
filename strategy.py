# 1d_KAMA_Trend_with_RSI_and_Volume_Confirmation_v1
# Hypothesis: Use daily KAMA for trend direction (adaptive moving average), daily RSI for momentum strength, and volume confirmation to filter noise.
# Long when KAMA is rising (trend up), RSI > 55 (bullish momentum), and volume > 1.5x 20-day average (confirmation).
# Short when KAMA is falling (trend down), RSI < 45 (bearish momentum), and volume > 1.5x 20-day average.
# Exit when trend reverses (KAMA direction change) or RSI reaches opposite extreme (40 for long exit, 60 for short exit).
# Uses weekly timeframe for trend filter: only take longs when weekly KAMA is rising, shorts when weekly KAMA is falling.
# Designed to work in both bull and bear markets by following the adaptive trend with momentum confirmation.
# Target: 15-25 trades/year by combining trend, momentum, and volume filters to reduce noise.

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
    
    # Get daily data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate daily KAMA ( Kaufman Adaptive Moving Average )
    kama_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, kama_period))  # |close(t) - close(t-kama_period)|
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=1)  # sum of absolute changes over kama_period
    # Need to handle array dimensions properly
    change_padded = np.concatenate([np.full(kama_period, np.nan), change])
    volatility_padded = np.concatenate([np.full(kama_period, np.nan), volatility])
    
    er = np.where(volatility_padded != 0, change_padded / volatility_padded, 0)
    
    # Smoothing constant
    sc = np.power(er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1), 2)
    
    # KAMA calculation
    kama_1d = np.full_like(close_1d, np.nan)
    kama_1d[kama_period] = close_1d[kama_period]  # Initialize
    
    for i in range(kama_period + 1, len(close_1d)):
        if not np.isnan(sc[i]):
            kama_1d[i] = kama_1d[i-1] + sc[i] * (close_1d[i] - kama_1d[i-1])
        else:
            kama_1d[i] = kama_1d[i-1]
    
    # Calculate daily RSI
    rsi_period = 14
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    
    # First average (simple mean)
    if len(close_1d) > rsi_period:
        avg_gain[rsi_period] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period] = np.mean(loss[:rsi_period])
        
        # Wilder smoothing
        for i in range(rsi_period + 1, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i-1]) / rsi_period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_1d = 100 - (100 / (1 + rs))
    else:
        rsi_1d = np.full_like(close_1d, np.nan)
    
    # Calculate weekly KAMA for trend filter
    if len(close_1w) >= kama_period:
        change_1w = np.abs(np.diff(close_1w, kama_period))
        volatility_1w = np.sum(np.abs(np.diff(close_1w)), axis=1)
        change_padded_1w = np.concatenate([np.full(kama_period, np.nan), change_1w])
        volatility_padded_1w = np.concatenate([np.full(kama_period, np.nan), volatility_1w])
        
        er_1w = np.where(volatility_padded_1w != 0, change_padded_1w / volatility_padded_1w, 0)
        sc_1w = np.power(er_1w * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1), 2)
        
        kama_1w = np.full_like(close_1w, np.nan)
        kama_1w[kama_period] = close_1w[kama_period]
        
        for i in range(kama_period + 1, len(close_1w)):
            if not np.isnan(sc_1w[i]):
                kama_1w[i] = kama_1w[i-1] + sc_1w[i] * (close_1w[i] - kama_1w[i-1])
            else:
                kama_1w[i] = kama_1w[i-1]
    else:
        kama_1w = np.full_like(close_1w, np.nan)
    
    # Volume confirmation: daily volume > 1.5x 20-day average
    vol_ma_period = 20
    vol_ma_1d = np.full_like(volume_1d, np.nan)
    
    if len(volume_1d) >= vol_ma_period:
        for i in range(vol_ma_period, len(volume_1d)):
            vol_ma_1d[i] = np.mean(volume_1d[i - vol_ma_period:i])
    
    # Align all daily indicators to 1d timeframe (no alignment needed as we're already on 1d)
    kama_1d_aligned = kama_1d
    rsi_1d_aligned = rsi_1d
    vol_ma_1d_aligned = vol_ma_1d
    
    # Align weekly KAMA to 1d timeframe
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(kama_period, rsi_period, vol_ma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(kama_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: rising or falling
        kama_rising = kama_1d_aligned[i] > kama_1d_aligned[i-1]
        kama_falling = kama_1d_aligned[i] < kama_1d_aligned[i-1]
        
        # Weekly trend filter
        weekly_kama_rising = kama_1w_aligned[i] > kama_1w_aligned[i-1]
        weekly_kama_falling = kama_1w_aligned[i] < kama_1w_aligned[i-1]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long: daily KAMA rising, weekly KAMA rising, RSI > 55, volume confirmation
            if kama_rising and weekly_kama_rising and rsi_1d_aligned[i] > 55 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: daily KAMA falling, weekly KAMA falling, RSI < 45, volume confirmation
            elif kama_falling and weekly_kama_falling and rsi_1d_aligned[i] < 45 and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA turns down OR RSI drops to 40
            if not kama_rising or rsi_1d_aligned[i] < 40:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA turns up OR RSI rises to 60
            if not kama_falling or rsi_1d_aligned[i] > 60:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_with_RSI_and_Volume_Confirmation_v1"
timeframe = "1d"
leverage = 1.0