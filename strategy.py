#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume confirmation
# Camarilla pivot levels identify institutional support/resistance zones
# Breakouts above R3 or below S3 with 12h trend alignment (EMA34) capture strong momentum
# Volume confirmation (1.8x 20-period average) filters false breakouts
# Works in bull/bear by only taking breakouts in direction of 12h EMA34 trend
# Discrete sizing 0.25 targets 80-180 trades over 4 years (20-45/year)

name = "4h_Camarilla_R3S3_Breakout_12hEMA34_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    # Calculate EMA(34) on 12h
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Camarilla pivot levels (R3, S3) from daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), etc.
    # S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    # Using previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla R3 and S3
    camarilla_r3 = prev_close + 1.125 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.125 * (prev_high - prev_low)
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price > Camarilla R3 with 12h uptrend (close > EMA34)
            long_breakout = close[i] > camarilla_r3_aligned[i]
            # Short breakdown: price < Camarilla S3 with 12h downtrend (close < EMA34)
            short_breakout = close[i] < camarilla_s3_aligned[i]
            
            # 12h EMA34 trend filter: close above/below EMA indicates trend direction
            ema_trend_up = close[i] > ema_34_12h_aligned[i]
            ema_trend_down = close[i] < ema_34_12h_aligned[i]
            
            if long_breakout and ema_trend_up and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif short_breakout and ema_trend_down and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < Camarilla S3 or trend reversal (close < EMA34)
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > Camarilla R3 or trend reversal (close > EMA34)
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals