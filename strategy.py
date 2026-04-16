#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data (HTF for trend) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate EMA34 on 12h
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate volume EMA34 on 12h
    vol_ema34_12h = pd.Series(volume_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    vol_ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ema34_12h)
    
    # === 1d data (HTF for support/resistance) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate ATR on 1d for stop loss
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 4h data (HTF for volatility filter) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate ATR on 4h for volatility filter
    tr_4h = np.maximum(high_4h - low_4h,
                       np.maximum(np.abs(high_4h - np.roll(close_4h, 1)),
                                  np.abs(low_4h - np.roll(close_4h, 1))))
    tr_4h[0] = high_4h[0] - low_4h[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # === Entry signals ===
    # Price above 12h EMA34 = uptrend
    # Price below 12h EMA34 = downtrend
    # Volume above 12h EMA34 = institutional interest
    # Low 4h volatility = good entry conditions
    
    # === 1h entry triggers ===
    # Buy when price closes above previous high AND volume spike
    # Sell when price closes below previous low AND volume spike
    
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ema34_12h_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_4h_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        ema34_12h_val = ema34_12h_aligned[i]
        vol_ema34_12h_val = vol_ema34_12h_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        atr_4h_val = atr_4h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below 12h EMA34 OR stop loss hit
            if (price < ema34_12h_val) or (price < entry_price - 1.5 * atr_1d_val):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above 12h EMA34 OR stop loss hit
            if (price > ema34_12h_val) or (price > entry_price + 1.5 * atr_1d_val):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session
            if in_session:
                # LONG: Price above 12h EMA34 (uptrend) AND volume above 12h EMA34 (institutional interest)
                # AND price breaks above previous high (momentum) AND low volatility
                if (price > ema34_12h_val) and (volume[i] > vol_ema34_12h_val) and \
                   (price > prev_high[i]) and (vol_ratio_val > 1.8) and (atr_4h_val < np.nanmedian(atr_4h_aligned[max(0, i-50):i+1])):
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                
                # SHORT: Price below 12h EMA34 (downtrend) AND volume above 12h EMA34 (institutional interest)
                # AND price breaks below previous low (momentum) AND low volatility
                elif (price < ema34_12h_val) and (volume[i] > vol_ema34_12h_val) and \
                     (price < prev_low[i]) and (vol_ratio_val > 1.8) and (atr_4h_val < np.nanmedian(atr_4h_aligned[max(0, i-50):i+1])):
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_EMA34_Volume_Momentum_LowVol_Session"
timeframe = "4h"
leverage = 1.0