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
    
    # === 6h data (primary for level calculation) ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # === 12h data (HTF for trend bias) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # === 1d data (HTF for volatility regime) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 6h ATR for volatility filter ===
    tr_6h = np.maximum(high_6h - low_6h,
                       np.maximum(np.abs(high_6h - np.roll(close_6h, 1)),
                                  np.abs(low_6h - np.roll(close_6h, 1))))
    tr_6h[0] = high_6h[0] - low_6h[0]
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    atr_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    
    # === 12h EMA for trend bias ===
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 1d Bollinger Band Width for volatility regime ===
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_band_1d = sma_20_1d + (2 * std_20_1d)
    lower_band_1d = sma_20_1d - (2 * std_20_1d)
    bb_width_1d = (upper_band_1d - lower_band_1d) / sma_20_1d
    bb_width_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_width_1d)
    
    # === 6h Donchian Channel (20) for entry levels ===
    highest_20_6h = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lowest_20_6h = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_upper_6h = highest_20_6h
    donchian_lower_6h = lowest_20_6h
    donchian_upper_6h_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper_6h)
    donchian_lower_6h_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower_6h)
    
    # === 6h Volume spike detection ===
    vol_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_6h = volume_6h / vol_ma_20_6h
    vol_ratio_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ratio_6h)
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour
    
    signals = np.zeros(n)
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_6h_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(bb_width_1d_aligned[i]) or np.isnan(donchian_upper_6h_aligned[i]) or 
            np.isnan(donchian_lower_6h_aligned[i]) or np.isnan(vol_ratio_6h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        atr_6h_val = atr_6h_aligned[i]
        ema_12h_val = ema_50_12h_aligned[i]
        bb_width_val = bb_width_1d_aligned[i]
        vol_ratio_val = vol_ratio_6h_aligned[i]
        upper_6h = donchian_upper_6h_aligned[i]
        lower_6h = donchian_lower_6h_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below 6h Donchian lower OR volatility regime shifts to high
            if (price < lower_6h) or (bb_width_val > 70):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above 6h Donchian upper OR volatility regime shifts to high
            if (price > upper_6h) or (bb_width_val > 70):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session
            if in_session:
                # LONG: Price breaks above 6h Donchian upper AND 12h EMA trend up AND low volatility AND volume spike
                if (price > upper_6h) and (price > ema_12h_val) and (bb_width_val < 30) and (vol_ratio_val > 1.5):
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Price breaks below 6h Donchian lower AND 12h EMA trend down AND low volatility AND volume spike
                elif (price < lower_6h) and (price < ema_12h_val) and (bb_width_val < 30) and (vol_ratio_val > 1.5):
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_12hEMA_Trend_DonchianBreakout_LowVol_Volume_Session"
timeframe = "6h"
leverage = 1.0