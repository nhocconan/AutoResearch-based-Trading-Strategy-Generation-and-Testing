#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and 1d volume spike filter
# Long when price breaks above R1 AND 4h close > EMA50 AND 1d volume > 1.8x 20-bar avg
# Short when price breaks below S1 AND 4h close < EMA50 AND 1d volume > 1.8x 20-bar avg
# Uses 4h/1d for signal direction, 1h only for entry timing precision
# Session filter: 08-20 UTC to avoid low-volume Asian session
# Target: 15-37 trades/year via tight entry conditions and multi-timeframe confluence
# Works in bull markets via breakouts with trend, in bear via mean reversion at S1/R1 in ranging markets

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h Camarilla pivot points (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12
    r1_1d = close_1d + camarilla_range
    s1_1d = close_1d - camarilla_range
    
    # Align Camarilla levels to 1h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 1d volume spike filter: >1.8x 20-bar average
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > 1.8 * volume_ma_20
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike, additional_delay_bars=0)
    
    # Session filter: 08-20 UTC (precomputed for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1h price data
    open_1h = prices['open'].values
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 30)  # Need sufficient history for HTF indicators
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 1h bar levels
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        vol_spike = volume_spike_aligned[i]
        ema_50 = ema_50_4h_aligned[i]
        close_price = close_1h[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above R1 AND 4h close > EMA50 (uptrend) AND volume spike
            if close_price > r1 and ema_50 > 0 and close_price > ema_50 and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below S1 AND 4h close < EMA50 (downtrend) AND volume spike
            elif close_price < s1 and ema_50 > 0 and close_price < ema_50 and vol_spike:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price breaks below S1 or trend changes
            if close_price < s1 or close_price < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short - exit when price breaks above R1 or trend changes
            if close_price > r1 or close_price > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals