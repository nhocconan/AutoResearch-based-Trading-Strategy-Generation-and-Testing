# 6h Multi-Timeframe Volume Spike + Weekly Trend Filter
# Long when price breaks above 6h VWAP + volume spike + weekly EMA21 uptrend
# Short when price breaks below 6h VWAP + volume spike + weekly EMA21 downtrend
# Exit when price returns to VWAP or weekly trend weakens
# Uses volume confirmation to filter breakouts and weekly EMA for trend alignment
# Designed for 6h timeframe: 50-150 total trades over 4 years (12-37/year)

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
    
    # Calculate 6h VWAP (typical price * volume) cumulative
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = vwap_numerator / vwap_denominator
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA21 for trend direction
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Get 6h data for volume average
    df_6h = get_htf_data(prices, '6h')
    volume_6h = df_6h['volume'].values
    
    # Volume average (20-period) on 6h for confirmation
    volume_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_20_6h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(vwap[i]) or np.isnan(ema_21_aligned[i]) or 
            np.isnan(volume_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        vwap_val = vwap[i]
        ema_21_val = ema_21_aligned[i]
        vol_ma = volume_ma_20_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for volume spikes with VWAP break and weekly trend alignment
            # Long: price breaks above VWAP + volume spike + weekly EMA21 rising
            if price > vwap_val and vol > 2.0 * vol_ma and ema_21_val > ema_21_aligned[i-1]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below VWAP + volume spike + weekly EMA21 falling
            elif price < vwap_val and vol > 2.0 * vol_ma and ema_21_val < ema_21_aligned[i-1]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit when price returns to VWAP or weekly trend turns down
            if price < vwap_val or ema_21_val < ema_21_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price returns to VWAP or weekly trend turns up
            if price > vwap_val or ema_21_val > ema_21_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_VWAP_VolumeSpike_WeeklyEMA21"
timeframe = "6h"
leverage = 1.0