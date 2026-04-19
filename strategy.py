#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and ATR volatility filter.
# Uses 1d EMA200 as long-term trend filter to avoid counter-trend entries.
# Works in bull/bear markets by filtering weak volatility and ensuring trend alignment.
# Target: 20-30 trades/year per symbol with controlled risk.
name = "4h_Donchian20_Volume_ATR_TrendFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate ATR (14-period) for volatility filter
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]), 
                       abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    # Donchian channels (20-period)
    def donchian_channels(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high, low, 20)
    
    # Align 1d EMA200 to 4h
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)  # Ensure EMA200 and Donchian are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_200_val = ema_200_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volatility filter: ATR > 0.5 * 20-period ATR average (avoid low volatility chop)
        atr_ma_20 = np.mean(atr[max(0, i-19):i+1]) if i >= 19 else atr_val
        vol_filter = atr_val > 0.5 * atr_ma_20
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Trend filter: price vs EMA200
        uptrend = price > ema_200_val
        downtrend = price < ema_200_val
        
        if position == 0:
            # Enter long on Donchian upper breakout with volume and trend confirmation
            if (price > donchian_upper[i] and volume_confirmed and 
                uptrend and vol_filter):
                signals[i] = 0.25
                position = 1
            # Enter short on Donchian lower breakdown with volume and trend confirmation
            elif (price < donchian_lower[i] and volume_confirmed and 
                  downtrend and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price crosses below Donchian lower or trend reverses
            if price < donchian_lower[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price crosses above Donchian upper or trend reverses
            if price > donchian_upper[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals