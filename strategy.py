#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA200 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high, price > EMA200, and volume spike.
# Short when price breaks below Donchian(20) low, price < EMA200, and volume spike.
# Uses ATR-based stoploss to limit drawdown. Designed for 4h timeframe with ~20-40 trades/year.
name = "4h_Donchian20_EMA200_Volume_Spike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # ATR(14) for stoploss
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]), 
                       abs(low[i] - close[i-1]))
        tr[0] = high[0] - low[0]
        
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    # Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        upper = np.zeros_like(high)
        lower = np.zeros_like(high)
        for i in range(len(high)):
            if i >= period - 1:
                upper[i] = np.max(high[i - period + 1:i + 1])
                lower[i] = np.min(low[i - period + 1:i + 1])
            else:
                upper[i] = np.nan
                lower[i] = np.nan
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d EMA200 to 4h
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)  # Ensure EMA200 and Donchian are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(donch_upper[i]) or 
            np.isnan(donch_lower[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_200_val = ema_200_aligned[i]
        upper = donch_upper[i]
        lower = donch_lower[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Enter long on Donchian breakout above upper band, price > EMA200, volume spike
            if price > upper and price > ema_200_val and volume_spike:
                signals[i] = 0.25
                position = 1
            # Enter short on Donchian breakdown below lower band, price < EMA200, volume spike
            elif price < lower and price < ema_200_val and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position: exit on Donchian breakdown below lower band or ATR stop
            if price < lower or price < ema_200_val - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short position: exit on Donchian breakout above upper band or ATR stop
            if price > upper or price > ema_200_val + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals