#!/usr/bin/env python3
"""
1d Donchian breakout with 1w EMA trend filter and volume confirmation.
Longs when price breaks above 20-day high with price > 1w EMA and volume > 1.5x average.
Shorts when price breaks below 20-day low with price < 1w EMA and volume > 1.5x average.
Exit on 2x ATR trailing stop or Donchian breakout in opposite direction.
Designed for 50-100 total trades over 4 years to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 10-period EMA on weekly close
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Daily high/low for Donchian channels (20-period)
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # 20-period Donchian channels
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume spike > 1.5x 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # ATR for trailing stop (20-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for trailing stop
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema_trend = ema_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: break above Donchian high with uptrend and volume
            if (price_high > upper and 
                price_close > ema_trend and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Enter short: break below Donchian low with downtrend and volume
            elif (price_low < lower and 
                  price_close < ema_trend and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Exit: 2x ATR trailing stop or opposite Donchian breakout
            exit_signal = False
            
            # ATR-based trailing stop
            if position == 1:
                # Long: trail from highest high since entry
                if price_close < entry_price - 2.0 * atr_val:
                    exit_signal = True
                # Update trailing stop if new high
                elif price_close > entry_price:
                    entry_price = price_close
            elif position == -1:
                # Short: trail from lowest low since entry
                if price_close > entry_price + 2.0 * atr_val:
                    exit_signal = True
                # Update trailing stop if new low
                elif price_close < entry_price:
                    entry_price = price_close
            
            # Opposite Donchian breakout exit
            if position == 1 and price_low < lower:
                exit_signal = True
            elif position == -1 and price_high > upper:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_1wEMA10_Volume1.5x_ATR2x_Trail"
timeframe = "1d"
leverage = 1.0