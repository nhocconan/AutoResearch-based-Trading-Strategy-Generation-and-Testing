#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_R1S1_Breakout_Volume_Control_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:  # Need at least 10 weeks for calculations
        return np.zeros(n)
    
    # === 1w: Calculate weekly trend using EMA34 ===
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === 1d: Calculate daily ATR for volatility filtering ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 1d: Calculate 1d OHLC for Camarilla pivot levels (using previous day's data) ===
    high_1d = high
    low_1d = low
    close_1d = close
    
    # Use previous day's OHLC for today's levels
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    
    # Set first day's values to NaN
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Calculate Camarilla levels: R1, S1
    # R1 = Close + (High - Low) * 1.1 / 12
    # S1 = Close - (High - Low) * 1.1 / 12
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # === 1d: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Get values
        close_val = close[i]
        ema34_val = ema34_1w_aligned[i]
        r1_level = camarilla_r1[i]
        s1_level = camarilla_s1[i]
        atr_val = atr[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema34_val) or np.isnan(r1_level) or np.isnan(s1_level) or 
            np.isnan(atr_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: require ATR > 0 (always true but keeps structure)
        vol_ok = atr_val > 0
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation and weekly uptrend
            if (close_val > r1_level and   # Break above R1
                vol_ratio_val > 2.0 and    # Strong volume confirmation
                close_val > ema34_val and  # Price above weekly EMA34 (uptrend)
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume confirmation and weekly downtrend
            elif (close_val < s1_level and   # Break below S1
                  vol_ratio_val > 2.0 and    # Strong volume confirmation
                  close_val < ema34_val and  # Price below weekly EMA34 (downtrend)
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price drops back below R1 (reversion to mean)
            if close_val < r1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises back above S1 (reversion to mean)
            if close_val > s1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals