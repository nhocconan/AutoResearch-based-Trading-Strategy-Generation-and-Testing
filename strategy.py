#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Supertrend_ETF_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d: Supertrend (ATR=10, multiplier=3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + (3 * atr_1d)
    lower_band = hl2 - (3 * atr_1d)
    
    # Final Bands and Trend
    final_upper = np.full(len(close_1d), np.nan)
    final_lower = np.full(len(close_1d), np.nan)
    supertrend = np.full(len(close_1d), np.nan)
    trend = np.full(len(close_1d), 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_1d)):
        if np.isnan(atr_1d[i]):
            continue
            
        # Upper Band
        if i == 1:
            final_upper[i] = upper_band[i]
            final_lower[i] = lower_band[i]
        else:
            if close_1d[i-1] <= final_upper[i-1]:
                final_upper[i] = upper_band[i]
            else:
                final_upper[i] = min(upper_band[i], final_upper[i-1])
                
            if close_1d[i-1] >= final_lower[i-1]:
                final_lower[i] = lower_band[i]
            else:
                final_lower[i] = max(lower_band[i], final_lower[i-1])
        
        # Supertrend and Trend
        if i == 1:
            if close_1d[i] > final_upper[i]:
                supertrend[i] = final_lower[i]
                trend[i] = -1
            else:
                supertrend[i] = final_upper[i]
                trend[i] = 1
        else:
            if supertrend[i-1] == final_upper[i-1]:
                if close_1d[i] <= final_upper[i]:
                    supertrend[i] = final_upper[i]
                    trend[i] = 1
                else:
                    supertrend[i] = final_lower[i]
                    trend[i] = -1
            else:
                if close_1d[i] >= final_lower[i]:
                    supertrend[i] = final_lower[i]
                    trend[i] = -1
                else:
                    supertrend[i] = final_upper[i]
                    trend[i] = 1
    
    # Align 1d Supertrend to 6h timeframe
    supertrend_1d = align_htf_to_ltf(prices, df_1d, supertrend)
    trend_1d = align_htf_to_ltf(prices, df_1d, trend)
    
    # === 6h: Price and volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 6h ATR for breakout threshold
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # EMA 20 for trend filter
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        st_level = supertrend_1d[i]
        trend_val = trend_1d[i]
        atr_val = atr[i]
        ema_val = ema_20[i]
        
        # Skip if any value is NaN
        if (np.isnan(st_level) or np.isnan(trend_val) or np.isnan(atr_val) or np.isnan(ema_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period median
        atr_median = np.nanmedian(atr[max(0, i-49):i+1]) if i >= 1 else np.nan
        vol_filter = atr_val > atr_median if not np.isnan(atr_median) else False
        
        if position == 0:
            # Long: Price breaks above Supertrend with volume confirmation and 1d uptrend
            if (close_val > st_level and   # Break above Supertrend
                vol_filter and             # Volatility filter
                trend_val == 1):           # 1d uptrend
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Supertrend with volume confirmation and 1d downtrend
            elif (close_val < st_level and   # Break below Supertrend
                  vol_filter and             # Volatility filter
                  trend_val == -1):          # 1d downtrend
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price drops back below Supertrend
            if close_val < st_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises back above Supertrend
            if close_val > st_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals