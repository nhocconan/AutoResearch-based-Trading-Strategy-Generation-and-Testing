#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 resistance in 1d uptrend (close > EMA200) with volume spike
# - Short when price breaks below Camarilla L3 support in 1d downtrend (close < EMA200) with volume spike
# - Uses ATR-based trailing stop: trail long at highest_high - 2.0*ATR, short at lowest_low + 2.0*ATR
# - Discrete position sizing (0.25) to minimize fee churn
# - Targets 20-40 trades/year (80-160 total over 4 years) to avoid fee drag
# - Works in bull via breakout continuation, in bear via mean reversion at extreme levels

name = "4h_1d_camarilla_breakout_volume_trend_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(200) for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 1d ATR(14) for stoploss calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = np.zeros_like(tr)
    atr_14_1d[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_1d[i] = (atr_14_1d[i-1] * (14-1) + tr[i]) / 14
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 1d volume confirmation: > 1.5x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 1d Camarilla pivot levels (based on previous day)
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 4
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_high = 0.0
    lowest_low = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high for trailing stop
            highest_high = max(highest_high, prices['high'].iloc[i])
            # Exit: ATR trailing stop or price breaks below L3 (failure)
            if (prices['close'].iloc[i] < highest_high - 2.0 * atr_14_1d_aligned[i] or 
                prices['close'].iloc[i] < camarilla_l3_aligned[i]):
                position = 0
                highest_high = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low for trailing stop
            lowest_low = min(lowest_low, prices['low'].iloc[i])
            # Exit: ATR trailing stop or price breaks above H3 (failure)
            if (prices['close'].iloc[i] > lowest_low + 2.0 * atr_14_1d_aligned[i] or 
                prices['close'].iloc[i] > camarilla_h3_aligned[i]):
                position = 0
                lowest_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla H3/L3 breakout with trend and volume filters
            if vol_spike_1d_aligned[i]:
                # Long signal: price breaks above H3 resistance in 1d uptrend
                if (prices['high'].iloc[i] >= camarilla_h3_aligned[i] and 
                    prices['close'].iloc[i] > ema_200_1d_aligned[i]):
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    highest_high = prices['high'].iloc[i]
                    signals[i] = 0.25
                # Short signal: price breaks below L3 support in 1d downtrend
                elif (prices['low'].iloc[i] <= camarilla_l3_aligned[i] and 
                      prices['close'].iloc[i] < ema_200_1d_aligned[i]):
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    lowest_low = prices['low'].iloc[i]
                    signals[i] = -0.25
    
    return signals