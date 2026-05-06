#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter, 1d volume spike, and session filter (08-20 UTC)
# Long when price breaks above R3 (strong resistance) AND 4h EMA50 uptrend AND 1d volume > 2.0 * 20-bar avg volume AND within session
# Short when price breaks below S3 (strong support) AND 4h EMA50 downtrend AND 1d volume > 2.0 * 20-bar avg volume AND within session
# Exit with signal=0 when price reverses back inside the Camarilla H-L range (mean reversion)
# Uses discrete sizing 0.20 to balance opportunity and drawdown
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Camarilla levels provide institutional pivot points; R3/S3 are strong breakout levels
# 4h EMA50 ensures higher-timeframe trend alignment to avoid counter-trend trades
# 1d volume spike confirms institutional participation
# Session filter reduces noise trades during low-liquidity hours
# Works in bull via buying strength on upside breakouts, works in bear via selling strength on downside breakdowns

name = "1h_Camarilla_R3S3_4hEMA50_1dVolumeSpike_Session_v1"
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
    
    # Pre-compute session filter (08-20 UTC) ONCE before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    close_4h_series = pd.Series(close_4h)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1h timeframe (wait for completed HTF bar)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data ONCE before loop for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume confirmation: volume > 2.0 * 20-bar average volume (stricter for fewer trades)
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * avg_volume_20_1d)
    
    # Align 1d volume spike to 1h timeframe
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate Camarilla levels from daily data (more stable than intraday)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: H-L range based
    R3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    S3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    H3 = close_1d + (high_1d - low_1d) * 1.1 / 6
    L3 = close_1d - (high_1d - low_1d) * 1.1 / 6
    
    # Align Camarilla levels to 1h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Camarilla breakout signals with trend, volume, and session filters
            # Long: price breaks above R3 (strong resistance) AND uptrend AND volume spike AND session
            if close[i] > R3_aligned[i] and close[i] > ema_50_4h_aligned[i] and volume_spike_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 (strong support) AND downtrend AND volume spike AND session
            elif close[i] < S3_aligned[i] and close[i] < ema_50_4h_aligned[i] and volume_spike_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price reverses back inside H3-L3 range (mean reversion)
            if close[i] < H3_aligned[i] and close[i] > L3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price reverses back inside H3-L3 range (mean reversion)
            if close[i] < H3_aligned[i] and close[i] > L3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals