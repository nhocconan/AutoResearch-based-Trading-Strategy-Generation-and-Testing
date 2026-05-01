#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses 4h for signal direction (trend) and 1d for regime filter (above/below EMA200)
# Entry: Price breaks Camarilla R3 (long) or S3 (short) on 1h with volume spike
# Trend: 4h EMA50 must align with breakout direction
# Regime: 1d close > EMA200 for longs, < EMA200 for shorts (avoid counter-trend in strong regimes)
# Session: 08-20 UTC to avoid low-liquidity hours
# Size: 0.20 (discrete, minimizes fee churn)
# Target: 15-30 trades/year (60-120 over 4 years) to avoid fee drag

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_1dEMA200_Regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h HTF data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d HTF data for EMA200 regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 1d EMA200 for regime filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate Camarilla levels for 1h (using previous bar's OHLC)
    # Camarilla: R4 = close + ((high-low)*1.1/2), R3 = close + ((high-low)*1.1/4)
    #          S3 = close - ((high-low)*1.1/4), S4 = close - ((high-low)*1.1/2)
    # We need previous bar's OHLC, so shift by 1
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # First bar has no previous
    
    rng = prev_high - prev_low
    camarilla_r3 = prev_close + (rng * 1.1 / 4)
    camarilla_s3 = prev_close - (rng * 1.1 / 4)
    
    # Volume confirmation: current volume > 1.5 * 24-period average volume
    volume_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (volume_ma_24 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, 200, 24)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(volume_ma_24[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        uptrend_4h = close[i] > ema_50_4h_aligned[i]
        downtrend_4h = close[i] < ema_50_4h_aligned[i]
        above_ema200_1d = close[i] > ema_200_1d_aligned[i]
        below_ema200_1d = close[i] < ema_200_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above R3, uptrend on 4h, above 1d EMA200, volume spike
            if close[i] > camarilla_r3[i] and uptrend_4h and above_ema200_1d and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below S3, downtrend on 4h, below 1d EMA200, volume spike
            elif close[i] < camarilla_s3[i] and downtrend_4h and below_ema200_1d and vol_spike:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on break below S3 or trend reversal
            if close[i] < camarilla_s3[i] or not uptrend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit on break above R3 or trend reversal
            if close[i] > camarilla_r3[i] or not downtrend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals