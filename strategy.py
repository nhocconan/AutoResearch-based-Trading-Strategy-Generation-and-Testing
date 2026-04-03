#!/usr/bin/env python3
"""
Experiment #107: 6h Camarilla Pivot + 1d Volume Spike + Regime Filter

HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) from 1d timeframe provide institutional support/resistance. 
Combined with 1d volume spikes (>2.0x average) to confirm institutional participation and a 6h ADX regime filter (ADX > 25 for trending, < 20 for ranging) 
to select appropriate strategy. In ranging markets (ADX < 20): fade at R3/S3 with target at R2/S2. In trending markets (ADX > 25): 
breakout continuation at R4/S4 with stoploss at R3/S3. This adapts to market conditions, working in both bull (breakouts with volume) 
and bear (failed reversals at pivots) markets. 6h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_107_6h_camarilla_1d_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot and volume (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from prior day's OHLC
    # Camarilla: 
    # R4 = Close + ((High - Low) * 1.1/2)
    # R3 = Close + ((High - Low) * 1.1/4)
    # R2 = Close + ((High - Low) * 1.1/6)
    # R1 = Close + ((High - Low) * 1.1/12)
    # PP = (High + Low + Close) / 3
    # S1 = Close - ((High - Low) * 1.1/12)
    # S2 = Close - ((High - Low) * 1.1/6)
    # S3 = Close - ((High - Low) * 1.1/4)
    # S4 = Close - ((High - Low) * 1.1/2)
    
    camarilla_r4 = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_r2 = np.full(n, np.nan)
    camarilla_r1 = np.full(n, np.nan)
    camarilla_pp = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    camarilla_s2 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    if len(df_1d) >= 2:  # Need at least 2 days of data
        # Align 1d data to LTF index for shifting
        df_1d_indexed = df_1d.set_index('open_time')
        
        # Calculate prior day's OHLC using shift(1) on the indexed series
        prior_day_high = df_1d_indexed['high'].shift(1).values
        prior_day_low = df_1d_indexed['low'].shift(1).values
        prior_day_close = df_1d_indexed['close'].shift(1).values
        
        # Calculate Camarilla levels for each prior day
        high_low_range = prior_day_high - prior_day_low
        
        camarilla_pp = (prior_day_high + prior_day_low + prior_day_close) / 3.0
        camarilla_r1 = prior_day_close + (high_low_range * 1.1 / 12)
        camarilla_s1 = prior_day_close - (high_low_range * 1.1 / 12)
        camarilla_r2 = prior_day_close + (high_low_range * 1.1 / 6)
        camarilla_s2 = prior_day_close - (high_low_range * 1.1 / 6)
        camarilla_r3 = prior_day_close + (high_low_range * 1.1 / 4)
        camarilla_s3 = prior_day_close - (high_low_range * 1.1 / 4)
        camarilla_r4 = prior_day_close + (high_low_range * 1.1 / 2)
        camarilla_s4 = prior_day_close - (high_low_range * 1.1 / 2)
        
        # Create series aligned with 1d index
        camarilla_levels = {
            'r4': camarilla_r4, 'r3': camarilla_r3, 'r2': camarilla_r2, 'r1': camarilla_r1,
            'pp': camarilla_pp, 's1': camarilla_s1, 's2': camarilla_s2, 's3': camarilla_s3, 's4': camarilla_s4
        }
        
        # Align each level to LTF (6h) timeframe with shift(1) for completed bars only
        camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
        camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
        camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
        camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
        camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
        camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
        camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
        camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
        camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    else:
        camarilla_r4_aligned = camarilla_r3_aligned = camarilla_r2_aligned = camarilla_r1_aligned = camarilla_pp_aligned = \
                               camarilla_s1_aligned = camarilla_s2_aligned = camarilla_s3_aligned = camarilla_s4_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
    df_1d_indexed = df_1d.set_index('open_time')
    prior_day_volume = df_1d_indexed['volume'].shift(1).values
    vol_ma_20 = pd.Series(prior_day_volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = prior_day_volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # === 6h Indicators: ADX(14) for regime detection ===
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Calculate Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Smoothed averages
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr_14
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 50  # Ensure enough data for HTF Camarilla, volume, and ADX
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(adx[i]) or np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: ADX > 25 = trending, ADX < 20 = ranging ---
        adx_trending = adx[i] > 25
        adx_ranging = adx[i] < 20
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_aligned[i] > 2.0
        
        # --- Price levels ---
        price = close[i]
        r4 = camarilla_r4_aligned[i]
        r3 = camarilla_r3_aligned[i]
        r2 = camarilla_r2_aligned[i]
        r1 = camarilla_r1_aligned[i]
        pp = camarilla_pp_aligned[i]
        s1 = camarilla_s1_aligned[i]
        s2 = camarilla_s2_aligned[i]
        s3 = camarilla_s3_aligned[i]
        s4 = camarilla_s4_aligned[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Determine exit based on regime
                if adx_ranging:  # Ranging: take profit at R2, stop at S3
                    if price >= r2 or price <= s3:
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
                else:  # Trending: trail stop at R3, take profit at R4
                    if price <= r3 or price >= r4:
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
            else:  # Short position
                # Determine exit based on regime
                if adx_ranging:  # Ranging: take profit at S2, stop at R3
                    if price <= s2 or price >= r3:
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
                else:  # Trending: trail stop at S3, take profit at S4
                    if price >= s3 or price <= s4:
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long conditions
        long_ranging = (adx_ranging and volume_spike and 
                       price <= s3 and price >= s4)  # Fade at S3/S4 in ranging
        long_trending = (adx_trending and volume_spike and 
                        price >= r4 and price <= r3)  # Breakout continuation at R4 in trending
        
        # Short conditions
        short_ranging = (adx_ranging and volume_spike and 
                        price >= r3 and price <= r4)  # Fade at R3/R4 in ranging
        short_trending = (adx_trending and volume_spike and 
                         price <= s4 and price >= s3)  # Breakdown continuation at S4 in trending
        
        if long_ranging or long_trending:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_ranging or short_trending:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals