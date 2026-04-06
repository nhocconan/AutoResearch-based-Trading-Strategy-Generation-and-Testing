#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour breakout with 4-hour ATR trend filter and volume confirmation.
# Uses 4-hour ATR to determine market volatility regime (high ATR = trending, low ATR = ranging).
# In trending regimes (high ATR): trade breakouts in direction of 1-day trend.
# In ranging regimes (low ATR): trade mean reversion at Bollinger Bands.
# Volume confirmation ensures institutional participation.
# Designed for 1h timeframe targeting 60-150 total trades over 4 years with proper risk management.
# Works in bull/bear markets via volatility regime adaptation.

name = "1h_atr_vol_breakout_meanrev_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4-hour ATR for volatility regime detection
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate True Range and ATR(14) on 4h
    tr_4h = np.zeros(len(close_4h))
    tr_4h[0] = high_4h[0] - low_4h[0]
    for i in range(1, len(close_4h)):
        tr_4h[i] = max(high_4h[i] - low_4h[i], 
                       abs(high_4h[i] - close_4h[i-1]), 
                       abs(low_4h[i] - close_4h[i-1]))
    
    atr_4h = np.full(len(close_4h), np.nan)
    if len(tr_4h) >= 14:
        atr_4h[13] = np.mean(tr_4h[:14])
        for i in range(14, len(tr_4h)):
            atr_4h[i] = (tr_4h[i] + 13 * atr_4h[i-1]) / 14
    
    # Align ATR to 1h timeframe (shifted by 1 4h bar for no look-ahead)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # 1-day trend via EMA200
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on 1d closes
    ema_200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema_200_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema_200_1d[i] = (close_1d[i] * 2 / 201) + (ema_200_1d[i-1] * 199 / 201)
    
    # Align EMA200 to 1h timeframe (shifted by 1 1d bar for no look-ahead)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 1-hour Bollinger Bands (20, 2)
    sma_20 = np.full(n, np.nan)
    std_20 = np.full(n, np.nan)
    
    for i in range(19, n):
        sma_20[i] = np.mean(close[i-19:i+1])
        std_20[i] = np.std(close[i-19:i+1])
    
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    # 1-hour volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    volume_filter = volume > vol_ma * 1.5
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available or outside session
        if (np.isnan(atr_4h_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(sma_20[i]) or np.isnan(std_20[i]) or 
            np.isnan(vol_ma[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volatility regime: high ATR = trending, low ATR = ranging
        # Using 50-period average of ATR to determine regime
        if i >= 50:
            atr_ma = np.mean(atr_4h_aligned[i-50:i+1])
            trending_regime = atr_4h_aligned[i] > atr_ma
        else:
            trending_regime = True  # default to trending for early bars
        
        # Check exits and stoploss (2x ATR)
        if position == 1:  # long position
            if i >= 14:  # need ATR for stop
                atr_val = atr_4h_aligned[i]
                if np.isnan(atr_val):
                    atr_val = np.nanmean(atr_4h_aligned[max(0, i-13):i+1])
                stop_loss = entry_price - 2.0 * atr_val
                
                if close[i] < stop_loss:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if i >= 14:
                atr_val = atr_4h_aligned[i]
                if np.isnan(atr_val):
                    atr_val = np.nanmean(atr_4h_aligned[max(0, i-13):i+1])
                stop_loss = entry_price + 2.0 * atr_val
                
                if close[i] > stop_loss:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
            else:
                signals[i] = -0.20
        else:
            # Look for entries based on regime
            if volume_filter[i]:
                if trending_regime:
                    # Trending regime: breakout in direction of 1d trend
                    bullish_trend = close[i] > ema_200_aligned[i]
                    bearish_trend = close[i] < ema_200_aligned[i]
                    
                    # Breakout conditions
                    breakout_up = high[i] > upper_band[i-1] if i > 0 else False
                    breakout_down = low[i] < lower_band[i-1] if i > 0 else False
                    
                    if breakout_up and bullish_trend:
                        signals[i] = 0.20
                        position = 1
                        entry_price = close[i]
                    elif breakout_down and bearish_trend:
                        signals[i] = -0.20
                        position = -1
                        entry_price = close[i]
                else:
                    # Ranging regime: mean reversion at Bollinger Bands
                    # Long at lower band, short at upper band
                    if close[i] <= lower_band[i]:
                        signals[i] = 0.20
                        position = 1
                        entry_price = close[i]
                    elif close[i] >= upper_band[i]:
                        signals[i] = -0.20
                        position = -1
                        entry_price = close[i]
    
    return signals