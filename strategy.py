#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend and volume spike
# Camarilla pivot levels (R3/S3) act as strong intraday support/resistance.
# Breakouts above R3 or below S3 with 1d EMA34 trend alignment capture momentum.
# Volume confirmation (2.0x 20-period average) filters false breakouts.
# Works in both bull/bear markets by only taking breakouts aligned with 1d EMA34.
# Discrete sizing 0.25 targets ~50-150 trades over 4 years (12-37/year).

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 1d data for EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day (using 1d OHLC)
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # We use previous 1d bar's OHLC to avoid look-ahead
    prev_1d_close = df_1d['close'].shift(1).values
    prev_1d_high = df_1d['high'].shift(1).values
    prev_1d_low = df_1d['low'].shift(1).values
    camarilla_r3 = prev_1d_close + 1.1 * (prev_1d_high - prev_1d_low) / 2
    camarilla_s3 = prev_1d_close - 1.1 * (prev_1d_high - prev_1d_low) / 2
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price > R3 with 1d uptrend (close > EMA34)
            long_breakout = close[i] > camarilla_r3_aligned[i]
            # Short breakdown: price < S3 with 1d downtrend (close < EMA34)
            short_breakout = close[i] < camarilla_s3_aligned[i]
            
            # 1d EMA34 trend filter: close above/below EMA indicates trend direction
            ema_trend_up = close[i] > ema_34_1d_aligned[i]
            ema_trend_down = close[i] < ema_34_1d_aligned[i]
            
            if long_breakout and ema_trend_up and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif short_breakout and ema_trend_down and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < S3 or trend reversal (close < EMA34)
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > R3 or trend reversal (close > EMA34)
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals