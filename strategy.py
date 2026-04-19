#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KC_Breakout_Volume_Kelly_v1"
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
    
    # Get daily data for Keltner Channel calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily high, low, close for ATR calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily True Range and ATR (20-period)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.absolute(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.absolute(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr_20_1d = pd.Series(tr1).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily EMA (20-period) for Keltner Channel middle
    ema_20_1d = pd.Series(close_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Calculate Keltner Channel bands (2 * ATR multiplier)
    upper_1d = ema_20_1d + 2 * atr_20_1d
    lower_1d = ema_20_1d - 2 * atr_20_1d
    
    # Align daily Keltner Channel levels to 4h timeframe
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average (4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Kelly criterion approximation: use volatility-adjusted position sizing
    # Base size scaled by inverse volatility (lower volatility = larger position)
    vol_normalized = atr_20_1d_aligned / np.nanmedian(atr_20_1d_aligned)
    kelly_scale = np.clip(1.0 / vol_normalized, 0.5, 2.0)  # Scale between 0.5x and 2x
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or 
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = upper_1d_aligned[i]
        lower = lower_1d_aligned[i]
        ema_mid = ema_20_1d_aligned[i]
        
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: break above upper KC band with volume
            if price > upper and volume_confirmed:
                base_size = 0.25
                adjusted_size = base_size * kelly_scale[i]
                signals[i] = np.clip(adjusted_size, 0.15, 0.35)  # Keep within reasonable bounds
                position = 1
            # Short: break below lower KC band with volume
            elif price < lower and volume_confirmed:
                base_size = -0.25
                adjusted_size = base_size * kelly_scale[i]
                signals[i] = np.clip(adjusted_size, -0.35, -0.15)  # Keep within reasonable bounds
                position = -1
        
        elif position == 1:
            # Exit: price below EMA middle or volatility expansion
            if price < ema_mid or vol > 3.0 * vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = np.clip(0.25 * kelly_scale[i], 0.15, 0.35)
        
        elif position == -1:
            # Exit: price above EMA middle or volatility expansion
            if price > ema_mid or vol > 3.0 * vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = np.clip(-0.25 * kelly_scale[i], -0.35, -0.15)
    
    return signals