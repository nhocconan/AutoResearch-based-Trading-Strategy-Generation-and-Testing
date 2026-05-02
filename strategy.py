#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike
# Uses 4h timeframe for signal generation with Camarilla pivot level breakouts
# 12h EMA(50) determines primary trend direction - multi-timeframe alignment
# Volume spike (2.0x 20-period average) ensures strong institutional participation
# Discrete position sizing (0.25) minimizes fee drag while maintaining profitability
# Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe
# Camarilla levels provide mathematically derived support/resistance based on prior day
# Works in both bull and bear markets by only taking trades aligned with 12h trend
# Prioritizes BTC/ETH over SOL by requiring volume confirmation and trend alignment

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_Volume_v1"
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
    open_time = prices['open_time'].values  # datetime64[ms]
    
    # Load 12h HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h EMA(50) for trend determination
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from prior 12h bar
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    #            S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We use R3/S3 for breakouts
    prior_close = np.roll(close_12h, 1)
    prior_high = np.roll(high_12h, 1)
    prior_low = np.roll(low_12h, 1)
    prior_close[0] = np.nan  # First value has no prior
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    
    camarilla_range = prior_high - prior_low
    camarilla_r3 = prior_close + 1.1 * camarilla_range
    camarilla_s3 = prior_close - 1.1 * camarilla_range
    
    # Align Camarilla levels to 4h timeframe (they update only when new 12h bar forms)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Close > Camarilla R3 + volume spike + close > 12h EMA50 (bullish trend)
            if close[i] > camarilla_r3_aligned[i] and volume_spike[i] and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close < Camarilla S3 + volume spike + close < 12h EMA50 (bearish trend)
            elif close[i] < camarilla_s3_aligned[i] and volume_spike[i] and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close < Camarilla S3 or close < 12h EMA50 (trend reversal)
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close > Camarilla R3 or close > 12h EMA50 (trend reversal)
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals