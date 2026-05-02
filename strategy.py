#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike + session filter (08-20 UTC)
# Uses 1h timeframe for signal generation with Camarilla pivot level breakouts
# 4h EMA(50) determines primary trend direction - multi-timeframe alignment with 4h trend
# Volume spike (2.0x 20-period average) ensures strong institutional participation
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
# Discrete position sizing (0.20) minimizes fee drag while maintaining profitability
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
# Camarilla levels provide mathematically derived support/resistance based on prior day
# Works in both bull and bear markets by only taking trades aligned with 4h trend
# Prioritizes BTC/ETH over SOL by requiring volume confirmation and trend alignment

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_Volume_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h EMA(50) for trend determination
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from prior 4h bar
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    #            S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We use R3/S3 for breakouts
    prior_close = np.roll(close_4h, 1)
    prior_high = np.roll(high_4h, 1)
    prior_low = np.roll(low_4h, 1)
    prior_close[0] = np.nan  # First value has no prior
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    
    camarilla_range = prior_high - prior_low
    camarilla_r3 = prior_close + 1.1 * camarilla_range
    camarilla_s3 = prior_close - 1.1 * camarilla_range
    
    # Align Camarilla levels to 1h timeframe (they update only when new 4h bar forms)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Close > Camarilla R3 + volume spike + close > 4h EMA50 (bullish trend)
            if close[i] > camarilla_r3_aligned[i] and volume_spike[i] and close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: Close < Camarilla S3 + volume spike + close < 4h EMA50 (bearish trend)
            elif close[i] < camarilla_s3_aligned[i] and volume_spike[i] and close[i] < ema_50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close < Camarilla S3 or close < 4h EMA50 (trend reversal)
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Close > Camarilla R3 or close > 4h EMA50 (trend reversal)
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals