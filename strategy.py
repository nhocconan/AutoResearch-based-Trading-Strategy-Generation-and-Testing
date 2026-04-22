#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian(20) breakout with 1d EMA trend filter,
# volume confirmation, and ATR-based stop loss. Designed to capture strong
# trending moves while avoiding false breakouts in low-volume or choppy conditions.
# The 1d EMA ensures alignment with higher timeframe trend, working in both bull
# and bear markets by only taking longs in uptrends and shorts in downtrends.
# Target: 20-50 trades per year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period high/low)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper and lower bands
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for stop loss and position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for volume confirmation
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_confirm = vol > 1.5 * vol_ma
        
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        ema_trend = ema_34_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high, above 1d EMA, with volume confirmation
            if price > donch_high_val and price > ema_trend and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below Donchian low, below 1d EMA, with volume confirmation
            elif price < donch_low_val and price < ema_trend and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # Exit conditions: stop loss or trend reversal
            exit_signal = False
            
            if position == 1:  # long position
                # Stop loss: 2 * ATR below entry price
                if price <= entry_price - 2.0 * atr_val:
                    exit_signal = True
                # Trend reversal: price crosses below 1d EMA
                elif price < ema_trend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Stop loss: 2 * ATR above entry price
                if price >= entry_price + 2.0 * atr_val:
                    exit_signal = True
                # Trend reversal: price crosses above 1d EMA
                elif price > ema_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_Trend_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0