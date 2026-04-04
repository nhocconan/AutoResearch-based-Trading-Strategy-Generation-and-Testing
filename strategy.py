#!/usr/bin/env python3
"""
Experiment #4511: 6h Elder Ray Index + 1d Regime Filter + Volume Confirmation
HYPOTHESIS: 6h Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) combined with 1d ADX regime filter (ADX>25 = trend, ADX<20 = range) and volume confirmation (>1.5x average) captures strong directional moves while avoiding choppy markets. In trending regimes (ADX>25), we trade Elder Ray extremes (Bull Power > 0 for long, Bear Power > 0 for short). In ranging regimes (ADX<20), we fade Elder Ray extremes at 2*std dev thresholds. This adaptive approach works in both bull and bear markets by aligning with the dominant 1d regime. Designed for 6h timeframe targeting 75-175 total trades over 4 years (19-44/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4511_6h_elder_ray_1d_regime_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 30:  # Need sufficient data for ADX calculation
        # Calculate ADX(14) on 1d data
        high_1d = pd.Series(df_1d['high'].values)
        low_1d = pd.Series(df_1d['low'].values)
        close_1d = pd.Series(df_1d['close'].values)
        
        # True Range
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        up_move = high_1d[1:] - high_1d[:-1]
        down_move = low_1d[:-1] - low_1d[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed values
        tr_ma = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
        plus_dm_ma = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
        minus_dm_ma = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_ma / tr_ma
        minus_di = 100 * minus_dm_ma / tr_ma
        
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Align ADX to 6h timeframe (shifted by 1 to avoid look-ahead)
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    else:
        adx_aligned = np.full(n, 25.0)  # Default to neutral trend
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # === 6h Indicators: Elder Ray Components ===
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(13, 20, 14, 30)  # EMA13, vol MA, ATR, 1d ADX data
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- Regime Determination ---
        is_trending = adx_aligned[i] > 25.0
        is_ranging = adx_aligned[i] < 20.0
        
        # --- Volume Confirmation ---
        volume_confirm = vol_ratio[i] > 1.5
        
        if is_trending:
            # In trending regimes: trade Elder Ray extremes in direction of power
            # Long when Bull Power > 0 (strong buying pressure)
            # Short when Bear Power > 0 (strong selling pressure)
            long_entry = bull_power[i] > 0 and volume_confirm
            short_entry = bear_power[i] > 0 and volume_confirm
        elif is_ranging:
            # In ranging regimes: fade Elder Ray extremes at 2*std dev
            # Calculate rolling statistics for Elder Ray
            if i >= 50:  # Need sufficient history for statistics
                bp_mean = np.nanmean(bull_power[max(0, i-50):i])
                bp_std = np.nanstd(bull_power[max(0, i-50):i])
                br_mean = np.nanmean(bear_power[max(0, i-50):i])
                br_std = np.nanstd(bear_power[max(0, i-50):i])
                
                # Long when Bear Power is extremely low (oversold)
                # Short when Bull Power is extremely high (overbought)
                long_entry = bear_power[i] < (br_mean - 2.0 * br_std) and volume_confirm
                short_entry = bull_power[i] > (bp_mean + 2.0 * bp_std) and volume_confirm
            else:
                long_entry = False
                short_entry = False
        else:
            # Transition zone (20 <= ADX <= 25): no trading to avoid whipsaw
            long_entry = False
            short_entry = False
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals