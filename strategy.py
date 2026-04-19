#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and ATR-based stoploss.
# Long when price breaks above Donchian(20) high with volume spike and price above 1d EMA200.
# Short when price breaks below Donchian(20) low with volume spike and price below 1d EMA200.
# Uses 1d EMA200 as trend filter to avoid counter-trend trades in bear markets.
# Volume confirmation ensures breakouts have institutional participation.
# ATR stoploss limits downside during volatile periods.
# Target: 20-50 trades/year per symbol (~80-200 total over 4 years).
name = "4h_Donchian20_1dEMA200_Volume_ATRStop"
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
    
    # Get daily data for EMA200
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily close
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA200 to 4h timeframe (wait for daily close)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Donchian channel (20-period) on 4h data
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR for volatility and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(200, 20, 14)  # Need EMA200, Donchian, and ATR data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        dc_high = high_max_20[i]
        dc_low = low_min_20[i]
        ema_trend = ema_200_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Enter long: price breaks above Donchian high AND above 1d EMA200
            if price > dc_high and price > ema_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Enter short: price breaks below Donchian low AND below 1d EMA200
            elif price < dc_low and price < ema_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions: stoploss or trend reversal
            # Stoploss: 2 * ATR below entry
            if price <= entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            # Or price breaks below Donchian low or below 1d EMA200
            elif price < dc_low or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: stoploss or trend reversal
            # Stoploss: 2 * ATR above entry
            if price >= entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            # Or price breaks above Donchian high or above 1d EMA200
            elif price > dc_high or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals