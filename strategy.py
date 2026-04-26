#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_DynamicRegime
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation (ATR ratio > 1.5). Uses dynamic position sizing based on volatility regime (ATR percentile) to reduce trades in choppy markets. Camarilla levels provide precise intraday support/resistance derived from prior day's range. Trend filter ensures alignment with higher timeframe momentum. Volume spike confirms institutional participation. Discrete sizing 0.25 limits trades (~20-40/year). Works in bull/bear via 1d trend filter and breakout logic.
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
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for volume regime and dynamic sizing
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period ATR) for volume regime
    atr_ratio = atr / pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate ATR percentile (20-period) for dynamic sizing - reduces size in high volatility
    atr_percentile = pd.Series(atr).rolling(window=20, min_periods=20).rank(pct=True).values
    
    # Calculate prior day's Camarilla levels from 1d OHLC
    # Camarilla levels: based on previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values  # previous day's high
    prev_low = df_1d['low'].shift(1).values    # previous day's low
    prev_close = df_1d['close'].shift(1).values # previous day's close
    
    # Camarilla R3, S3 levels (most significant for breakouts)
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    camarilla_range = prev_high - prev_low
    r3 = prev_close + camarilla_range * 1.1 / 4
    s3 = prev_close - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (each 1d bar = 6x 4h bars)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Base position size
    base_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (34 for EMA, 50 for ATR ratio, 1 for shift)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_ratio[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(atr_percentile[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = atr_ratio[i] > 1.5  # volume spike confirmation
        
        # Dynamic sizing: reduce size in high volatility (percentile > 0.8)
        volatility_multiplier = 0.5 if atr_percentile[i] > 0.8 else 1.0
        size = base_size * volatility_multiplier
        
        # Entry conditions: Camarilla R3/S3 breakout with volume spike AND aligned with 1d EMA34 trend
        # Long: price breaks above R3 (bullish breakout)
        # Short: price breaks below S3 (bearish breakout)
        long_entry = (close_val > r3_aligned[i]) and vol_spike and (close_val > ema_34_val)
        short_entry = (close_val < s3_aligned[i]) and vol_spike and (close_val < ema_34_val)
        
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
            # Long - exit when price re-enters Camarilla H3/L3 range or trend reversal
            # H3 = Close + (High - Low) * 1.1/2
            # L3 = Close - (High - Low) * 1.1/2
            h3 = prev_close[i] + camarilla_range[i] * 1.1 / 2 if not (np.isnan(prev_close[i]) or np.isnan(camarilla_range[i])) else np.nan
            l3 = prev_close[i] - camarilla_range[i] * 1.1 / 2 if not (np.isnan(prev_close[i]) or np.isnan(camarilla_range[i])) else np.nan
            h3_aligned = align_htf_to_ltf(prices, df_1d, h3) if not np.isnan(h3) else np.full(n, np.nan)
            l3_aligned = align_htf_to_ltf(prices, df_1d, l3) if not np.isnan(l3) else np.full(n, np.nan)
            
            h3_val = h3_aligned[i] if i < len(h3_aligned) else np.nan
            l3_val = l3_aligned[i] if i < len(l3_aligned) else np.nan
            
            if (not np.isnan(h3_val) and not np.isnan(l3_val) and 
                close_val < h3_val and close_val > l3_val):  # back inside H3/L3 range
                signals[i] = 0.0
                position = 0
            elif close_val < ema_34_val:  # trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price re-enters Camarilla H3/L3 range or trend reversal
            h3 = prev_close[i] + camarilla_range[i] * 1.1 / 2 if not (np.isnan(prev_close[i]) or np.isnan(camarilla_range[i])) else np.nan
            l3 = prev_close[i] - camarilla_range[i] * 1.1 / 2 if not (np.isnan(prev_close[i]) or np.isnan(camarilla_range[i])) else np.nan
            h3_aligned = align_htf_to_ltf(prices, df_1d, h3) if not np.isnan(h3) else np.full(n, np.nan)
            l3_aligned = align_htf_to_ltf(prices, df_1d, l3) if not np.isnan(l3) else np.full(n, np.nan)
            
            h3_val = h3_aligned[i] if i < len(h3_aligned) else np.nan
            l3_val = l3_aligned[i] if i < len(l3_aligned) else np.nan
            
            if (not np.isnan(h3_val) and not np.isnan(l3_val) and 
                close_val > l3_val and close_val < h3_val):  # back inside H3/L3 range
                signals[i] = 0.0
                position = 0
            elif close_val > ema_34_val:  # trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_DynamicRegime"
timeframe = "4h"
leverage = 1.0