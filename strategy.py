#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume spike and 1w trend filter
# - Long when price breaks above R3 (Camarilla resistance) AND 1d volume > 2.0x 20-period volume SMA AND price > 1w EMA(50)
# - Short when price breaks below S3 (Camarilla support) AND 1d volume > 2.0x 20-period volume SMA AND price < 1w EMA(50)
# - Exit: price returns to 1d Camarilla pivot point (PP) or opposing breakout
# - Uses 1d for Camarilla levels and volume confirmation, 1w for trend filter, 6h for price action
# - Volume spike ensures institutional participation; trend filter avoids counter-trend trades
# - Camarilla levels provide precise support/resistance based on prior day's range
# - Works in both bull and bear markets as breakouts occur in all regimes with volume confirmation

name = "6h_1d_1w_camarilla_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate 1d Camarilla levels from prior day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on prior day's range
    rng = high_1d - low_1d
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r3 = camarilla_pp + rng * 1.1 / 4.0
    camarilla_s3 = camarilla_pp - rng * 1.1 / 4.0
    camarilla_r4 = camarilla_pp + rng * 1.1 / 2.0
    camarilla_s4 = camarilla_pp - rng * 1.1 / 2.0
    
    # Align 1d Camarilla levels to 6h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 1d volume SMA for confirmation
    vol_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(20, n):  # Start from 20 to have sufficient lookback for indicators
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 2.0x 20-period volume SMA
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_confirm = vol_1d_aligned[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: price relative to 1w EMA(50)
        trend_filter_long = close[i] > ema_50_1w_aligned[i]
        trend_filter_short = close[i] < ema_50_1w_aligned[i]
        
        # Only trade when volume confirmation is present
        if vol_confirm:
            # Long breakout: price breaks above R3 AND trend filter long
            if close[i] > camarilla_r3_aligned[i] and trend_filter_long:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25  # Maintain position
            # Short breakout: price breaks below S3 AND trend filter short
            elif close[i] < camarilla_s3_aligned[i] and trend_filter_short:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25  # Maintain position
            # Exit: price returns to pivot point or breaks opposite level
            elif (abs(close[i] - camarilla_pp_aligned[i]) < (camarilla_r4_aligned[i] - camarilla_s4_aligned[i]) * 0.02) or \
                 (position == 1 and close[i] < camarilla_s3_aligned[i]) or \
                 (position == -1 and close[i] > camarilla_r3_aligned[i]):
                if position != 0:  # Only signal on exit
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.0  # Maintain flat
            else:
                # Maintain current position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # No trade: exit any position if conditions not met
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals