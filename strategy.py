# Solution: 12h_Camarilla_R1S1_Breakout_Volume_ATRFilter_V1
# Hypothesis: Camarilla pivot levels (R1, S1) from 1d timeframe act as support/resistance.
# Price breaking above R1 with volume > 1.5x average and ATR > 0.5% of price indicates bullish momentum.
# Price breaking below S1 with volume > 1.5x average and ATR > 0.5% of price indicates bearish momentum.
# Position size 0.25 for risk control. Works in both bull (buy breakouts) and bear (sell breakdowns).
# Uses 12h timeframe to reduce trade frequency and avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # === 1d data (higher timeframe for Camarilla pivot levels) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Calculate Camarilla pivot levels (R1, S1) from 1d data ===
    # Pivot point = (High + Low + Close) / 3
    # R1 = Close + 1.1 * (High - Low) / 12
    # S1 = Close - 1.1 * (High - Low) / 12
    pp_1d = (high_1d + low_1d + close_1d) / 3
    r1_1d = close_1d + 1.1 * (high_1d - low_1d) / 12
    s1_1d = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 12h ATR for volatility filter ===
    tr_12h = np.maximum(
        high_12h - low_12h,
        np.maximum(
            np.abs(high_12h - np.roll(close_12h, 1)),
            np.abs(low_12h - np.roll(close_12h, 1))
        )
    )
    tr_12h[0] = high_12h[0] - low_12h[0]  # First value
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # === 12h volume ratio for confirmation ===
    vol_ma_10_12h = pd.Series(volume_12h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_12h = volume_12h / vol_ma_10_12h
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(atr_12h[i]) or np.isnan(vol_ratio_12h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1 = r1_12h[i]
        s1 = s1_12h[i]
        atr_val = atr_12h[i]
        vol_ratio = vol_ratio_12h[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            if price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            if price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price breaks below S1
            if price < s1:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above R1
            if price > r1:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require sufficient volatility (ATR > 0.5% of price)
            if atr_val > 0.005 * price:
                # Buy when price breaks above R1 with volume confirmation
                if price > r1 and vol_ratio > 1.5:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Sell when price breaks below S1 with volume confirmation
                elif price < s1 and vol_ratio > 1.5:
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

name = "12h_Camarilla_R1S1_Breakout_Volume_ATRFilter_V1"
timeframe = "12h"
leverage = 1.0