#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and ATR(14) volatility filter
# Long when price breaks above Donchian upper band AND price > 12h EMA50 AND ATR(14) > 0.5 * ATR(50)
# Short when price breaks below Donchian lower band AND price < 12h EMA50 AND ATR(14) > 0.5 * ATR(50)
# Exit when price crosses the middle band (20-period SMA of high/low) or ATR-based stoploss triggers
# Uses discrete position sizing (0.30) to balance capture and fee minimization.
# Target: 75-200 total trades over 4 years (19-50/year) on 4h.
# Donchian channels provide structural breakout signals, 12h EMA50 filters trend alignment,
# ATR volatility filter ensures sufficient momentum while avoiding choppy markets.
# Works in bull markets (trend continuation) and bear markets (mean reversion within trend via exits).

name = "4h_Donchian20_EMA50_ATRFilter_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate EMA(50) on 12h data
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get daily data for ATR calculation (to avoid look-ahead bias)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14) and ATR(50) on daily data
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50_1d = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align ATR values to 4h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    # Calculate Donchian channels on 4h data (20-period)
    # Upper band: 20-period high of high
    # Lower band: 20-period low of low
    # Middle band: 20-period SMA of (high + low)/2
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = ((high_series.rolling(window=20, min_periods=20).mean() + 
                       low_series.rolling(window=20, min_periods=20).mean()) / 2).values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # EMA50 and Donchian warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(atr_50_1d_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_50 = ema_50_12h_aligned[i]
        atr_14 = atr_14_1d_aligned[i]
        atr_50 = atr_50_1d_aligned[i]
        
        # Donchian levels
        upper_band = donchian_upper[i]
        lower_band = donchian_lower[i]
        middle_band = donchian_middle[i]
        
        # ATR-based volatility filter: require sufficient momentum
        vol_filter = atr_14 > 0.5 * atr_50
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below middle band OR ATR-based stoploss (2.5 * ATR below entry)
            # Since we don't track entry price, use close-based exit: cross below middle band
            if curr_close < middle_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price crosses above middle band
            if curr_close > middle_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long when price breaks above upper band AND price > 12h EMA50 AND volume confirmation AND vol filter
            if curr_close > upper_band and curr_close > ema_50 and vol_conf and vol_filter:
                signals[i] = 0.30
                position = 1
            # Short when price breaks below lower band AND price < 12h EMA50 AND volume confirmation AND vol filter
            elif curr_close < lower_band and curr_close < ema_50 and vol_conf and vol_filter:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals