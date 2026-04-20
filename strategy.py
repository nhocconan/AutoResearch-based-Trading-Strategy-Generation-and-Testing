#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_R1S1_Breakout_Volume_Control_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === 1d: Calculate daily OHLC ===
    open_d = prices['open'].values
    high_d = prices['high'].values
    low_d = prices['low'].values
    close_d = prices['close'].values
    volume_d = prices['volume'].values
    
    # === 1w: Calculate weekly trend using EMA34 ===
    # Weekly close from 1w data
    close_1w = df_1w['close'].values
    # Calculate EMA34 on weekly close
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align weekly EMA to daily timeframe (wait for weekly bar to close)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 1d: Calculate 1d ATR for volatility filter ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 1d: Calculate 1d volume ratio (current vs 20-period average) ===
    vol_ma20 = pd.Series(volume_d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_d / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === 1d: Calculate daily Camarilla pivot levels (using previous day's data) ===
    # Use previous day's OHLC for today's levels
    prev_close = np.roll(close_d, 1)
    prev_high = np.roll(high_d, 1)
    prev_low = np.roll(low_d, 1)
    
    # Set first day's values to NaN
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Calculate Camarilla levels: R1, S1
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = close_d[i]
        r1_level = camarilla_r1[i]
        s1_level = camarilla_s1[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        weekly_trend = ema_34_1w_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_level) or np.isnan(s1_level) or np.isnan(vol_ratio_val) or 
            np.isnan(atr_val) or np.isnan(weekly_trend)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period median (avoid choppy markets)
        atr_median = np.nanmedian(atr[max(0, i-49):i+1])
        vol_filter = atr_val > atr_median if not np.isnan(atr_median) else False
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation, weekly uptrend, and volatility filter
            if (close_val > r1_level and   # Break above R1
                vol_ratio_val > 2.0 and    # Strong volume confirmation
                weekly_trend > close_val and  # Weekly uptrend (price below EMA34)
                vol_filter):               # Volatility filter
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume confirmation, weekly downtrend, and volatility filter
            elif (close_val < s1_level and   # Break below S1
                  vol_ratio_val > 2.0 and    # Strong volume confirmation
                  weekly_trend < close_val and  # Weekly downtrend (price above EMA34)
                  vol_filter):               # Volatility filter
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price drops back below R1 (reversion to mean) or weekly trend turns down
            if close_val < r1_level or weekly_trend < close_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises back above S1 (reversion to mean) or weekly trend turns up
            if close_val > s1_level or weekly_trend > close_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals