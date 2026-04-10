#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long when price breaks above H3 (bullish bias) AND 1w EMA(20) > EMA(50) (bullish trend) AND 12h volume > 1.5x 20-bar avg
# - Short when price breaks below L3 (bearish bias) AND 1w EMA(20) < EMA(50) (bearish trend) AND 12h volume > 1.5x 20-bar avg
# - Exit when price returns to H4/L4 levels (Camarilla pivot extremes) or opposite pivot level
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Camarilla pivots provide precise intraday support/resistance levels that work well on 12h timeframe
# - 1w EMA filter ensures alignment with weekly trend to avoid counter-trend trades
# - Volume confirmation avoids low-liquidity false signals
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: pivot breaks in ranges, trend filter prevents counter-trend whipsaws

name = "12h_1w_camarilla_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA trend filter: EMA(20) vs EMA(50)
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_bullish_1w = ema_20_1w > ema_50_1w
    ema_bearish_1w = ema_20_1w < ema_50_1w
    
    # Align 1w EMA trend to 12h timeframe
    ema_bullish_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_bullish_1w)
    ema_bearish_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_bearish_1w)
    
    # Pre-compute 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Calculate Camarilla levels
    range_1d = high_1d - low_1d
    h3 = pivot + (range_1d * 1.1 / 4)
    l3 = pivot - (range_1d * 1.1 / 4)
    h4 = pivot + (range_1d * 1.1 / 2)
    l4 = pivot - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Pre-compute 12h volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_1w_aligned[i]) or np.isnan(ema_bearish_1w_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(vol_spike[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        price = prices['close'].iloc[i]
        
        if position == 0:  # Flat - look for new Camarilla breakout entries
            # Long when price breaks above H3 AND 1w bullish trend AND volume spike
            if (price > h3_aligned[i] and 
                ema_bullish_1w_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L3 AND 1w bearish trend AND volume spike
            elif (price < l3_aligned[i] and 
                  ema_bearish_1w_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price returns to H4/L4 levels or crosses opposite pivot level
            exit_signal = False
            if position == 1:  # Long position
                # Exit when price crosses below L4 (opposite extreme) or reaches H4 (target)
                if price < l4_aligned[i] or price > h4_aligned[i]:
                    exit_signal = True
            else:  # Short position
                # Exit when price crosses above H4 (opposite extreme) or reaches L4 (target)
                if price > h4_aligned[i] or price < l4_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals