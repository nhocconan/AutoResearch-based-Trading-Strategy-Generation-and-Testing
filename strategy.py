#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation
# Uses actual Camarilla pivot calculation from 1d OHLC: H4 = C + 1.5*(H-L), L4 = C - 1.5*(H-L)
# R3 = H4 + 1.5*(H4-L4), S3 = L4 - 1.5*(H4-L4) where H4/L4 are Camarilla pivot levels
# Long when price breaks above R3 with volume > 2x average AND price > 1w EMA50
# Short when price breaks below S3 with volume > 2x average AND price < 1w EMA50
# Discrete sizing 0.25 to limit fee drag, targeting 20-40 trades/year

name = "4h_Camarilla_R3S3_Breakout_1wEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)  # warmup for indicators
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema_1w = ema_50_1w_aligned[i]
        
        # Get 1d data for Camarilla calculation (completed 1d bar)
        if i >= 96:  # Need at least 96 4h bars (4*24) for 1d lookback
            # Get index of completed 1d bar (look back 96 4h bars = 24 1d bars)
            idx_1d = (i // 96) * 96
            if idx_1d >= 96 and idx_1d + 96 <= len(high):
                # Use the 1d bar that completed 96 4h bars ago (to avoid look-ahead)
                start_1d = idx_1d - 96
                end_1d = idx_1d
                if end_1d <= len(high) and start_1d >= 0:
                    # Calculate Camarilla levels from the completed 1d bar
                    day_high = np.max(high[start_1d:end_1d])
                    day_low = np.min(low[start_1d:end_1d])
                    day_close = close[end_1d-1]  # Close of completed 1d bar
                    
                    # Camarilla pivot calculation
                    range_val = day_high - day_low
                    if range_val > 0:
                        H4 = day_close + 1.5 * range_val
                        L4 = day_close - 1.5 * range_val
                        R3 = H4 + 1.5 * (H4 - L4)
                        S3 = L4 - 1.5 * (H4 - L4)
                        
                        # Volume confirmation: current volume > 2x 20-period average
                        if i >= 20:
                            vol_ma_20 = np.mean(volume[i-20:i])
                        else:
                            vol_ma_20 = 0.0
                        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
                        
                        # Handle exits and stoploss
                        if position == 1:  # Long position
                            # Exit: price breaks below H4 (Camarilla H4 level) OR stoploss
                            stop_price = curr_close - 2.0 * (curr_high - curr_low)  # ATR approximation
                            if curr_close < H4 or curr_close < stop_price:
                                signals[i] = 0.0
                                position = 0
                            else:
                                signals[i] = 0.25
                                
                        elif position == -1:  # Short position
                            # Exit: price breaks above L4 (Camarilla L4 level) OR stoploss
                            stop_price = curr_close + 2.0 * (curr_high - curr_low)  # ATR approximation
                            if curr_close > L4 or curr_close > stop_price:
                                signals[i] = 0.0
                                position = 0
                            else:
                                signals[i] = -0.25
                                
                        else:  # Flat - look for new entries
                            # Long entry: price breaks above R3 with volume spike AND price > 1w EMA50
                            if (curr_close > R3 and 
                                vol_spike and 
                                curr_close > curr_ema_1w):
                                signals[i] = 0.25
                                position = 1
                            # Short entry: price breaks below S3 with volume spike AND price < 1w EMA50
                            elif (curr_close < S3 and 
                                  vol_spike and 
                                  curr_close < curr_ema_1w):
                                signals[i] = -0.25
                                position = -1
                            else:
                                signals[i] = 0.0
                        continue
        
        # Default flat if conditions not met
        signals[i] = 0.0
    
    return signals