#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Daily Camarilla R3/S3 level breakouts with weekly trend filter (price above/below 1w EMA34) and volume confirmation (ATR ratio > 1.3). Camarilla levels provide high-probability reversal/breakout points. Weekly trend ensures alignment with higher timeframe momentum to avoid counter-trend trades. Volume spike confirms institutional participation. Discrete sizing 0.25 targets ~20-40 trades/year. Works in bull/bear via weekly trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR(14) for volume regime
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period ATR) for volume regime
    atr_ratio = atr / pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    # Camarilla equations:
    # H4 = Close + 1.1*(High-Low)/2
    # L4 = Close - 1.1*(High-Low)/2
    # H3 = Close + 1.1*(High-Low)/4
    # L3 = Close - 1.1*(High-Low)/4
    # H2 = Close + 1.1*(High-Low)/6
    # L2 = Close - 1.1*(High-Low)/6
    # H1 = Close + 1.1*(High-Low)/12
    # L1 = Close - 1.1*(High-Low)/12
    # R3 = H3, S3 = L3 (we focus on these levels)
    
    # Shift OHLC by 1 to use previous day's data for today's levels
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    
    # Calculate Camarilla levels
    hl_range = prev_high - prev_low
    camarilla_h3 = prev_close + 1.1 * hl_range / 4
    camarilla_l3 = prev_close - 1.1 * hl_range / 4
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (34 for weekly EMA, 50 for ATR ratio)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        h3_val = camarilla_h3[i]
        l3_val = camarilla_l3[i]
        ema_34_val = ema_34_1w_aligned[i]
        vol_spike = atr_ratio[i] > 1.3  # volume regime
        size = fixed_size
        
        # Entry conditions: Camarilla R3/S3 breakout with volume spike AND aligned with weekly EMA34 trend
        # Long: price breaks above R3 (bullish breakout)
        # Short: price breaks below S3 (bearish breakout)
        long_entry = (close_val > h3_val) and vol_spike and (close_val > ema_34_val)
        short_entry = (close_val < l3_val) and vol_spike and (close_val < ema_34_val)
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price re-enters Camarilla H3-L3 range or trend reversal
            if close_val < h3_val and close_val > l3_val:  # back inside H3-L3 range
                signals[i] = 0.0
                position = 0
            elif close_val < ema_34_val:  # trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price re-enters Camarilla H3-L3 range or trend reversal
            if close_val > l3_val and close_val < h3_val:  # back inside H3-L3 range
                signals[i] = 0.0
                position = 0
            elif close_val > ema_34_val:  # trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0