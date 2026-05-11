# 4h_Three_Stage_Filter_Signal
# Hypothesis: 4h price breaks above/below 1d VWAP, confirmed by 1d trend (EMA50) and volume spike (2x 4h VWMA20).
# Uses multi-timeframe structure: 1d for trend and VWAP reference, 4h for entry timing and volume confirmation.
# Designed for low trade frequency (<50/year) to minimize fee drag while capturing strong moves.
# Works in bull/bear by following 1d trend direction.

name = "4h_Three_Stage_Filter_Signal"
timeframe = "4h"
leverage = 1.0

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
    
    # === 1D Data for Trend Filter and VWAP Reference ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1D EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Previous 1D VWAP (typical price * volume / cumulative volume)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_num = np.cumsum(typical_price_1d * volume_1d)
    vwap_den = np.cumsum(volume_1d)
    vwap_1d = vwap_num / vwap_den
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # === 4H Volume Confirmation: 2.0x VWMA20 ===
    typical_price_4h = (high + low + close) / 3.0
    vwap_num_4h = np.cumsum(typical_price_4h * volume)
    vwap_den_4h = np.cumsum(volume)
    vwap_4h = vwap_num_4h / vwap_den_4h
    
    # VWMA20: volume-weighted moving average of price
    vwma_num = np.convolve(typical_price_4h * volume, np.ones(20), 'full')[:len(typical_price_4h)]
    vwma_den = np.convolve(volume, np.ones(20), 'full')[:len(volume)]
    vwma20 = np.divide(vwma_num, vwma_den, out=np.full_like(vwma_den, np.nan), where=vwma_den!=0)
    
    volume_spike = volume > (vwma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers 1D EMA50)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vwma20[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above 1D VWAP with uptrend and volume spike
            if (close[i] > vwap_1d_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below 1D VWAP with downtrend and volume spike
            elif (close[i] < vwap_1d_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below 1D VWAP
            if close[i] < vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses back above 1D VWAP
            if close[i] > vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals