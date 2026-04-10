#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation
# - Long when price breaks above H3 level with volume > 1.3x average AND daily close > daily EMA50
# - Short when price breaks below L3 level with volume > 1.3x average AND daily close < daily EMA50
# - Exit when price retests pivot point (PP) or volume drops below average
# - Daily trend filter ensures alignment with major trend
# - Volume confirmation prevents false breakouts
# - Camarilla levels work well in ranging markets (common in bearish 2025+)
# - Targets 12-30 trades/year (50-120 total over 4 years) to avoid fee drag
# - Uses discrete position sizes (0.0, ±0.25) to minimize fee churn

name = "12h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute Camarilla pivot levels from previous 1d bar
    # H4 = C + 1.5*(H-L), H3 = C + 1.0*(H-L), H2 = C + 0.75*(H-L), H1 = C + 0.5*(H-L)
    # L4 = C - 1.5*(H-L), L3 = C - 1.0*(H-L), L2 = C - 0.75*(H-L), L1 = C - 0.5*(H-L)
    # PP = (H + L + C) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels for previous day (to avoid look-ahead)
    PP_1d = (high_1d + low_1d + close_1d) / 3.0
    H3_1d = close_1d + 1.0 * (high_1d - low_1d)
    L3_1d = close_1d - 1.0 * (high_1d - low_1d)
    
    # Align to 12h timeframe (wait for completed 1d bar)
    PP_1d_aligned = align_htf_to_ltf(prices, df_1d, PP_1d)
    H3_1d_aligned = align_htf_to_ltf(prices, df_1d, H3_1d)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
    
    # Pre-compute 1d EMA(50) for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute volume confirmation: > 1.3x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.3 * volume_20_avg)
    
    # Pre-compute volume filter: < average volume for exit
    vol_normal = prices['volume'] < volume_20_avg
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(PP_1d_aligned[i]) or np.isnan(H3_1d_aligned[i]) or 
            np.isnan(L3_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > H3 with volume spike AND daily uptrend
            if (prices['high'].iloc[i] > H3_1d_aligned[i] and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price < L3 with volume spike AND daily downtrend
            elif (prices['low'].iloc[i] < L3_1d_aligned[i] and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retests pivot point (mean reversion signal)
            # 2. Volume drops below average (loss of momentum)
            if position == 1:  # Long position
                if (prices['low'].iloc[i] <= PP_1d_aligned[i] or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (prices['high'].iloc[i] >= PP_1d_aligned[i] or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals