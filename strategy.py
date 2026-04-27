# 12h_Camilla_R1S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot breakout on 12h with 1d trend filter and volume confirmation
# Works in bull/bear by following trend direction while using volatility-based entries
# Target: 50-150 trades over 4 years (12-37/year) with low frequency to avoid fee drag

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
    
    # Get 1d data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA 34 for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d ATR for volatility filter
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 12h Camarilla levels (based on previous 12h bar)
    # Calculate pivot and levels from previous bar
    shift_high = np.roll(high, 1)
    shift_low = np.roll(low, 1)
    shift_close = np.roll(close, 1)
    shift_high[0] = np.nan
    shift_low[0] = np.nan
    shift_close[0] = np.nan
    
    pivot = (shift_high + shift_low + shift_close) / 3.0
    range_val = shift_high - shift_low
    
    # Camarilla levels
    R1 = pivot + (range_val * 1.0 / 12.0)
    S1 = pivot - (range_val * 1.0 / 12.0)
    R3 = pivot + (range_val * 1.1 / 12.0)
    S3 = pivot - (range_val * 1.1 / 12.0)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Volatility filter: current 12h ATR > 1.2x 1d ATR (avoid low volatility periods)
        atr_12h = np.abs(high[i] - low[i])
        vol_filter = atr_12h > (atr_1d_aligned[i] * 1.2)
        
        # Long conditions: price breaks above R1 + above 1d EMA + volume + volatility
        long_breakout = (close[i] > R1[i-1] and price_above_ema and volume_filter[i] and vol_filter)
        # Short conditions: price breaks below S1 + below 1d EMA + volume + volatility
        short_breakout = (close[i] < S1[i-1] and price_below_ema and volume_filter[i] and vol_filter)
        
        if long_breakout:
            signals[i] = 0.25
            position = 1
        elif short_breakout:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite S3/R3 breakout (wider bands for exit)
        elif position == 1 and close[i] < S3[i-1]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > R3[i-1]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0