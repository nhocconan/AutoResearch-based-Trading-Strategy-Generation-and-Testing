#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with 1d EMA34 trend filter and volume confirmation.
# Camarilla levels (S1/S2/R1/R2) from prior 1d candle provide high-probability reversal zones.
# Enter long at S1 with bullish rejection (close > open), short at R1 with bearish rejection (close < open).
# Filter by 1d EMA34 trend: only long when price > EMA34, short when price < EMA34.
# Volume spike (>1.5x 20-period average) confirms institutional interest.
# Designed for low trade frequency (~20-35/year) to minimize fee decay.
# Works in bull/bear by trading reversals within the dominant trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla and EMA (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from prior 1d candle
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), R2 = close + 0.6*(high-low)
    # R1 = close + 0.382*(high-low), S1 = close - 0.382*(high-low)
    # S2 = close - 0.6*(high-low), S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    rng = high_1d - low_1d
    r1 = close_1d + 0.382 * rng
    s1 = close_1d - 0.382 * rng
    
    # Calculate 34-period EMA on 1d close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 4h timeframe (waits for 1d bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        open_price = prices['open'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_34_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        # Price rejection at levels: bullish at S1 (close > open), bearish at R1 (close < open)
        bullish_rejection = (price > s1_val) and (open_price <= s1_val) and (price > open_price)
        bearish_rejection = (price < r1_val) and (open_price >= r1_val) and (price < open_price)
        
        if position == 0:
            # Long conditions: bullish rejection at S1 + uptrend + volume spike
            if bullish_rejection and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish rejection at R1 + downtrend + volume spike
            elif bearish_rejection and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to opposite level or trend breaks
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price reaches R1 or trend breaks
                if price >= r1_val or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price reaches S1 or trend breaks
                if price <= s1_val or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_S1R1_Reversal_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0