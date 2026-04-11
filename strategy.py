#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return signals
    
    # Calculate 4h ATR for volatility filter (14-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_14_4h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Calculate 4h ATR MA (20-period) for volatility regime filter
    atr_ma_20_4h = pd.Series(atr_14_4h_aligned).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h ATR ratio (current / MA) for regime detection
    atr_ratio_4h = np.where(atr_ma_20_4h > 0, atr_14_4h_aligned / atr_ma_20_4h, 1.0)
    
    # Calculate Camarilla levels on 4h data (using previous 4h bar's range)
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    prev_close_4h[0] = np.nan
    
    camarilla_H4_4h = prev_close_4h + 1.1 * (prev_high_4h - prev_low_4h) / 2
    camarilla_L4_4h = prev_close_4h - 1.1 * (prev_high_4h - prev_low_4h) / 2
    camarilla_C_4h = prev_close_4h  # Pivot point
    
    # Align Camarilla levels to 1h timeframe
    camarilla_H4_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_H4_4h)
    camarilla_L4_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_L4_4h)
    camarilla_C_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_C_4h)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Calculate 1d EMA(200) for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: 24-period average on 1h (1 day)
    volume_sma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_H4_4h_aligned[i]) or np.isnan(camarilla_L4_4h_aligned[i]) or
            np.isnan(camarilla_C_4h_aligned[i]) or np.isnan(volume_sma_24[i]) or
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(atr_ratio_4h[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        hour = hours[i]
        
        # Session filter: trade only between 08:00-20:00 UTC
        in_session = (8 <= hour <= 20)
        
        # Volume confirmation: current volume > 1.5x 24-period average
        vol_confirm = volume_current > 1.5 * volume_sma_24[i]
        
        # Trend filter: price above/below 1d EMA200
        price_above_ema200 = price_close > ema_200_1d_aligned[i]
        price_below_ema200 = price_close < ema_200_1d_aligned[i]
        
        # Volatility filter: avoid choppy markets (ATR ratio < 0.8 or > 1.2)
        vol_filter = (atr_ratio_4h[i] >= 0.8) and (atr_ratio_4h[i] <= 1.2)
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price breaks above Camarilla H4 level + volume confirmation + uptrend + volatility filter + session
        price_above_H4 = price_close > camarilla_H4_4h_aligned[i]
        if price_above_H4 and vol_confirm and price_above_ema200 and vol_filter and in_session:
            enter_long = True
        
        # Short: Price breaks below Camarilla L4 level + volume confirmation + downtrend + volatility filter + session
        price_below_L4 = price_close < camarilla_L4_4h_aligned[i]
        if price_below_L4 and vol_confirm and price_below_ema200 and vol_filter and in_session:
            enter_short = True
        
        # Exit conditions: price crosses back through the Camarilla C level (pivot)
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price crosses below Camarilla C level
            exit_long = price_close < camarilla_C_4h_aligned[i]
        elif position == -1:
            # Exit short if price crosses above Camarilla C level
            exit_short = price_close > camarilla_C_4h_aligned[i]
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.20
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 1h Camarilla breakout with 4h/1d filters for institutional-grade entries.
# Uses 4h Camarilla levels (H4/L4) for entry/exit and 1d EMA200 for trend filter.
# Volume confirmation (>1.5x 24-period average) ensures strong participation.
# Volatility filter (ATR ratio between 0.8-1.2) avoids choppy markets.
# Session filter (08-20 UTC) focuses on liquid trading hours.
# Position size: 0.20 to manage risk. Target: 15-30 trades/year to minimize fee drag.
# Works in bull markets (trend filter captures uptrends) and bear markets (short signals in downtrends).