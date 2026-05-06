#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h price action filtered by 4h trend (Supertrend) and 1d momentum (RSI)
# Uses 4h Supertrend for trend direction, 1d RSI for momentum strength, and volume spike for confirmation
# Enters on 1h pullbacks to the Supertrend in the direction of the higher timeframe trend
# Designed for low trade frequency (15-30/year) to avoid fee drag, works in bull/bear via trend following
# Static position size 0.20 to manage drawdown; exits via opposite signal or trend reversal

name = "1h_Supertrend4h_RSI1d_VolumeFilter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h Supertrend (ATR=10, mult=3.0) for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(10)
    atr_4h = pd.Series(tr_4h).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high_4h + low_4h) / 2
    upper_band = hl2 + 3.0 * atr_4h
    lower_band = hl2 - 3.0 * atr_4h
    
    # Initialize arrays
    upper_band_final = np.full_like(upper_band, np.nan)
    lower_band_final = np.full_like(lower_band, np.nan)
    supertrend = np.full_like(close_4h, np.nan)
    trend = np.ones_like(close_4h, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    # First valid value
    start_idx = 10
    if len(close_4h) > start_idx:
        upper_band_final[start_idx] = upper_band[start_idx]
        lower_band_final[start_idx] = lower_band[start_idx]
        supertrend[start_idx] = upper_band_final[start_idx]
        trend[start_idx] = 1
        
        for i in range(start_idx + 1, len(close_4h)):
            # Upper band
            if upper_band[i] < upper_band_final[i-1] or close_4h[i-1] > upper_band_final[i-1]:
                upper_band_final[i] = upper_band[i]
            else:
                upper_band_final[i] = upper_band_final[i-1]
            
            # Lower band
            if lower_band[i] > lower_band_final[i-1] or close_4h[i-1] < lower_band_final[i-1]:
                lower_band_final[i] = lower_band[i]
            else:
                lower_band_final[i] = lower_band_final[i-1]
            
            # Trend
            if trend[i-1] == 1 and close_4h[i] <= lower_band_final[i]:
                trend[i] = -1
                supertrend[i] = upper_band_final[i]
            elif trend[i-1] == -1 and close_4h[i] >= upper_band_final[i]:
                trend[i] = 1
                supertrend[i] = lower_band_final[i]
            elif trend[i-1] == 1:
                trend[i] = 1
                supertrend[i] = lower_band_final[i]
            else:
                trend[i] = -1
                supertrend[i] = upper_band_final[i]
    
    # Align Supertrend trend to 1h
    supertrend_trend_4h = trend  # 1 for uptrend, -1 for downtrend
    supertrend_trend_1h = align_htf_to_ltf(prices, df_4h, supertrend_trend_4h)
    
    # Calculate 1d RSI(14) for momentum strength
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    avg_gain[13] = np.nanmean(gain[1:14])
    avg_loss[13] = np.nanmean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = np.concatenate([[np.nan] * 14, rsi_1d])
    
    # Align RSI to 1h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume spike filter (>2.0x 24-bar average)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (2.0 * vol_ma_24)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(supertrend_trend_1h[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: 4h uptrend, 1d RSI > 50 (bullish momentum), volume spike
            if supertrend_trend_1h[i] == 1 and rsi_1d_aligned[i] > 50 and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short entry: 4h downtrend, 1d RSI < 50 (bearish momentum), volume spike
            elif supertrend_trend_1h[i] == -1 and rsi_1d_aligned[i] < 50 and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: 4h trend turns down OR RSI turns bearish
            if supertrend_trend_1h[i] == -1 or rsi_1d_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: 4h trend turns up OR RSI turns bullish
            if supertrend_trend_1h[i] == 1 or rsi_1d_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals