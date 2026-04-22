#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA trend filter and volume spike
# Long when price breaks above R1, 12h EMA50 rising, volume > 2x 20-period average
# Short when price breaks below S1, 12h EMA50 falling, volume > 2x 20-period average
# Exit when price returns to Camarilla Pivot point or 12h EMA trend reverses
# Camarilla levels provide precise intraday support/resistance, EMA50 filters trend direction,
# volume spike confirms breakout strength. Designed for low trade frequency (~20-40/year)
# with edge in both trending and ranging markets through volatility expansion breakouts.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (based on previous day)
    # R4 = close + ((high - low) * 1.5000)
    # R3 = close + ((high - low) * 1.2500)
    # R2 = close + ((high - low) * 1.1666)
    # R1 = close + ((high - low) * 1.0833)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.0833)
    # S2 = close - ((high - low) * 1.1666)
    # S3 = close - ((high - low) * 1.2500)
    # S4 = close - ((high - low) * 1.5000)
    
    # Shift by 1 to use previous day's levels
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = prev_high[1] if len(prev_high) > 1 else prev_high[0]
    prev_low[0] = prev_low[1] if len(prev_low) > 1 else prev_low[0]
    prev_close[0] = prev_close[1] if len(prev_close) > 1 else prev_close[0]
    
    camarilla_r1 = prev_close + ((prev_high - prev_low) * 1.0833)
    camarilla_s1 = prev_close - ((prev_high - prev_low) * 1.0833)
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    
    # Load 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        pp = camarilla_pp_aligned[i]
        ema = ema_50_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        # EMA trend: rising if current > previous, falling if current < previous
        ema_rising = ema > ema_50_aligned[i-1] if i > 0 else False
        ema_falling = ema < ema_50_aligned[i-1] if i > 0 else False
        
        if position == 0:
            # Long conditions: price breaks above R1, EMA rising, volume spike
            if price > r1 and ema_rising and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S1, EMA falling, volume spike
            elif price < s1 and ema_falling and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to pivot or EMA trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to pivot or EMA starts falling
                if price <= pp or not ema_rising:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to pivot or EMA starts rising
                if price >= pp or not ema_falling:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0