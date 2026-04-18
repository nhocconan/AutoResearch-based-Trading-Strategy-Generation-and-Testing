# 4h_1dCCI_Reversal_WithVolumeFilter
# Hypothesis: Mean-reversion strategy using daily CCI to identify overbought/oversold conditions. 
# Enter long when 1d CCI crosses below -100 (oversold) with price above 4h EMA34 (bullish bias) and volume > 1.5x average.
# Enter short when 1d CCI crosses above +100 (overbought) with price below 4h EMA34 (bearish bias) and volume > 1.5x average.
# Exit on opposite CCI signal. Uses CCI's mean-reversion tendency in ranging markets and EMA filter to avoid counter-trend trades.
# Designed for low trade frequency (<30/year) to minimize fee drag while capturing reversals in both bull and bear markets.

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
    
    # Get 1d data for CCI
    df_1d = get_htf_data(prices, '1d')
    
    # 1d CCI(20)
    cci_period = 20
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    
    cci_1d = np.full_like(typical_price_1d, np.nan)
    
    if len(typical_price_1d) >= cci_period:
        # Calculate moving average of typical price
        ma_tp = np.full_like(typical_price_1d, np.nan)
        for i in range(cci_period - 1, len(typical_price_1d)):
            ma_tp[i] = np.mean(typical_price_1d[i - cci_period + 1:i + 1])
        
        # Calculate mean deviation
        md = np.full_like(typical_price_1d, np.nan)
        for i in range(cci_period - 1, len(typical_price_1d)):
            dev = np.abs(typical_price_1d[i - cci_period + 1:i + 1] - ma_tp[i])
            md[i] = np.mean(dev)
        
        # Calculate CCI
        cci_1d = (typical_price_1d - ma_tp) / (0.015 * md)
        # Handle division by zero
        cci_1d = np.where(md == 0, 0, cci_1d)
    
    # Align 1d CCI to 4h timeframe
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d)
    
    # EMA34 on 4h for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    ema_period = 34
    ema_4h = np.full_like(close_4h, np.nan)
    
    if len(close_4h) >= ema_period:
        multiplier = 2 / (ema_period + 1)
        ema_4h[ema_period - 1] = np.mean(close_4h[:ema_period])
        for i in range(ema_period, len(close_4h)):
            ema_4h[i] = (close_4h[i] * multiplier) + (ema_4h[i-1] * (1 - multiplier))
    
    # Align EMA to 4h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(cci_period, ema_period, vol_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(cci_1d_aligned[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # CCI crossover signals
        cci_long_signal = (i > 0 and not np.isnan(cci_1d_aligned[i-1]) and 
                          cci_1d_aligned[i-1] >= -100 and cci_1d_aligned[i] < -100)
        cci_short_signal = (i > 0 and not np.isnan(cci_1d_aligned[i-1]) and 
                           cci_1d_aligned[i-1] <= 100 and cci_1d_aligned[i] > 100)
        
        if position == 0:
            # Long: CCI crosses below -100 (oversold) + price above EMA34 + volume
            if cci_long_signal and close[i] > ema_4h_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: CCI crosses above +100 (overbought) + price below EMA34 + volume
            elif cci_short_signal and close[i] < ema_4h_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: CCI crosses above +100 (overbought) or price below EMA34
            if (cci_short_signal or close[i] < ema_4h_aligned[i]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CCI crosses below -100 (oversold) or price above EMA34
            if (cci_long_signal or close[i] > ema_4h_aligned[i]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dCCI_Reversal_WithVolumeFilter"
timeframe = "4h"
leverage = 1.0