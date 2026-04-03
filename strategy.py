#!/usr/bin/env python3
"""
Experiment #219: 6h Camarilla Pivot + Volume Spike + Chop Filter

HYPOTHESIS: Camarilla pivot levels (R3/S3, R4/S4) derived from 1d timeframe act as 
intraday support/resistance. Price rejecting at R3/S3 with volume confirmation indicates 
mean reversion opportunities, while breaking R4/S4 with volume indicates continuation 
trends. Choppiness filter (CHOP > 61.8 = ranging) avoids false signals in choppy markets. 
6h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag 
while capturing significant pivot reactions. Works in both bull (continuation breaks) 
and bear (mean reversion at pivots) markets through adaptive logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_219_6h_camarilla_pivot_volume_chop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from prior 1d OHLC
    # Prior day's OHLC
    prior_day_high = df_1d['high'].shift(1).values
    prior_day_low = df_1d['low'].shift(1).values
    prior_day_close = df_1d['close'].shift(1).values
    
    # Camarilla levels
    # R4 = Close + ((High - Low) * 1.1/2)
    # R3 = Close + ((High - Low) * 1.1/4)
    # S3 = Close - ((High - Low) * 1.1/4)
    # S4 = Close - ((High - Low) * 1.1/2)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    if len(df_1d) >= 2:  # Need at least 2 days of data (current + prior)
        high_low_range = prior_day_high - prior_day_low
        camarilla_r4 = prior_day_close + (high_low_range * 1.1 / 2.0)
        camarilla_r3 = prior_day_close + (high_low_range * 1.1 / 4.0)
        camarilla_s3 = prior_day_close - (high_low_range * 1.1 / 4.0)
        camarilla_s4 = prior_day_close - (high_low_range * 1.1 / 2.0)
    
    # Align Camarilla levels to 6h timeframe with shift(1) for completed bars only
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === 6h Indicators: Choppiness Index (14) for regime filter ===
    atr_14_chop = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14_chop).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.full(n, np.nan)
    denominator = np.log10(14) * (highest_high_14 - lowest_low_14)
    chop[13:] = 100 * np.log10(sum_atr_14[13:] / np.where(denominator[13:] != 0, denominator[13:], 1))
    chop[:13] = 50.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Ensure enough data for HTF Camarilla, ATR, and chop
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # === Camarilla Logic ===
        # Price above R4: bullish continuation breakout
        # Price between R3 and R4: mean reversion zone (fade R3/R4)
        # Price between S3 and S4: mean reversion zone (fade S3/S4)
        # Price below S4: bearish continuation breakdown
        
        price_above_r4 = close[i] > camarilla_r4_aligned[i]
        price_below_s4 = close[i] < camarilla_s4_aligned[i]
        price_between_r3_r4 = (close[i] > camarilla_r3_aligned[i]) & (close[i] < camarilla_r4_aligned[i])
        price_between_s3_s4 = (close[i] > camarilla_s3_aligned[i]) & (close[i] < camarilla_s4_aligned[i])
        
        # === Volume Confirmation: Require volume spike (> 1.8x average) ===
        volume_spike = vol_ratio[i] > 1.8
        
        # === Chop Filter: Avoid signals in excessively choppy markets (CHOP > 61.8) ===
        chop_filter = chop[i] <= 61.8  # Only allow signals when not excessively choppy
        
        # === Exit Logic (ATR-based stoploss) ===
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite Camarilla level
                if close[i] < camarilla_s3_aligned[i]:  # Long TP at S3
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite Camarilla level
                if close[i] > camarilla_r3_aligned[i]:  # Short TP at R3
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # === New Position Entry Logic (Only if Flat) ===
        # Long: 
        #   - Continuation: break above R4 with volume
        #   - Mean reversion: rejection at S3 with volume (price < S3 but reverting)
        long_continuation = price_above_r4 and volume_spike and chop_filter
        long_mean_reversion = (close[i] < camarilla_s3_aligned[i]) and volume_spike and chop_filter and (close[i] > open[i])  # Bullish candle near S3
        
        # Short:
        #   - Continuation: break below S4 with volume
        #   - Mean reversion: rejection at R3 with volume (price > R3 but reverting)
        short_continuation = price_below_s4 and volume_spike and chop_filter
        short_mean_reversion = (close[i] > camarilla_r3_aligned[i]) and volume_spike and chop_filter and (close[i] < open[i])  # Bearish candle near R3
        
        if long_continuation or long_mean_reversion:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_continuation or short_mean_reversion:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals