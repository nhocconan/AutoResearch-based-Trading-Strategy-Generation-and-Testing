#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels provide precise intraday support/resistance. Breakouts above R3 or below S3
# indicate strong momentum. Trend filter ensures we trade with the 1d EMA34 direction. Volume spike
# confirms breakout validity. Designed to capture strong trends while avoiding false breakouts.
# Target: 20-50 trades/year (80-200 total over 4 years) on BTC/ETH/SOL.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels for 4h (based on previous bar's range)
    # Camarilla: R4 = close + 1.1*(high-low)/2, R3 = close + 1.1*(high-low)/4,
    #            S3 = close - 1.1*(high-low)/4, S4 = close - 1.1*(high-low)/2
    # We use previous bar's high/low to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # First bar has no previous
    
    rng = prev_high - prev_low
    R3 = prev_close + 1.1 * rng / 4
    S3 = prev_close - 1.1 * rng / 4
    
    # Volume confirmation (2.0x 20-period average) on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Camarilla and EMA)
    start_idx = 34  # max(20 for volume MA, 34 for EMA)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Close above R3 + 1d uptrend + volume spike
            if close[i] > R3[i] and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Close below S3 + 1d downtrend + volume spike
            elif close[i] < S3[i] and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below S3 (reversal) or trend reversal
            if close[i] < S3[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above R3 (reversal) or trend reversal
            if close[i] > R3[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals