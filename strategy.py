#!/usr/bin/env python3
"""
Experiment #2247: 6h Camarilla Pivot Reversion + 1d Trend Filter + Volume Spike
HYPOTHESIS: On 6h timeframe, price tends to revert from extreme Camarilla levels (R4/S4, R3/S3) 
when aligned with 1d trend and confirmed by volume spikes. Works in bull/bear via trend filter.
- Primary: 6h Camarilla pivot levels from prior 1d, fade at R3/S3, breakout at R4/S4
- HTF: 1d EMA(50) trend filter (only trade in direction of daily trend)
- Entry: Volume > 2.0x 20-bar average + price at Camarilla extreme levels
- Exit: Opposite Camarilla level touch or ATR(14) stop (2*ATR)
- Target: 75-150 total trades over 4 years (19-37/year) - optimized for 6h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2247_6h_camarilla_pivot_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 6h Indicators: Camarilla Pivots from prior 1d, Volume MA(20), ATR(14) ===
    # Calculate Camarilla levels using prior 1d OHLC (loaded via HTF)
    # Need to align prior day's OHLC to current 6h bars
    df_1d = get_htf_data(prices, '1d')  # Already called above, but get again for clarity in calculation
    # Actually, we'll compute Camarilla using the 1d data we already have
    
    # Extract prior day's OHLC for each 6h bar
    # We need to shift the 1d data by 1 to get prior completed day
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to get prior day's values (lookback 1 completed day)
    prior_open = np.roll(open_1d, 1)
    prior_high = np.roll(high_1d, 1)
    prior_low = np.roll(low_1d, 1)
    prior_close = np.roll(close_1d, 1)
    prior_open[0] = np.nan
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # Calculate Camarilla levels for prior day
    # Camarilla: 
    # H = prior_high, L = prior_low, C = prior_close
    # R4 = C + (H-L)*1.1/2
    # R3 = C + (H-L)*1.1/4
    # S3 = C - (H-L)*1.1/4
    # S4 = C - (H-L)*1.1/2
    H = prior_high
    L = prior_low
    C = prior_close
    range_hl = H - L
    
    camarilla_r4 = C + range_hl * 1.1 / 2
    camarilla_r3 = C + range_hl * 1.1 / 4
    camarilla_s3 = C - range_hl * 1.1 / 4
    camarilla_s4 = C - range_hl * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2*ATR below highest since entry
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches S3 (mean reversion target)
                elif price <= camarilla_s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2*ATR above lowest since entry
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches R3 (mean reversion target)
                elif price >= camarilla_r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment for bias filter
        trend_bias = trend_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Fade extreme levels: sell at R3/R4, buy at S3/S4
            # Short entry: price at or above R3 AND 1d trend down (fade rallies in downtrend)
            if trend_bias < 0 and price >= camarilla_r3_aligned[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            # Long entry: price at or below S3 AND 1d trend up (buy dips in uptrend)
            elif trend_bias > 0 and price <= camarilla_s3_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals