#!/usr/bin/env python3
"""
Experiment #3259: 6h Camarilla Pivot + 12h ADX Trend + Volume Spike
HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) combined with 12h ADX trend filter and volume confirmation.
In ranging markets (ADX < 25): fade extreme Camarilla levels (R3/S3) for mean reversion.
In trending markets (ADX > 25): breakout continuation at R4/S4 levels.
Volume spike (>1.5x 20-period average) confirms institutional participation.
Designed to work in both bull (trend continuation) and bear (mean reversion from extremes) markets by adapting to regime.
Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3259_6h_camarilla_pivot_12h_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for ADX trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX(14) on 12h data
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, adjust=False).mean().values
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        # DX and ADX
        dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
        adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === 6h Indicators: Camarilla Pivot Levels (based on previous day) ===
    # Camarilla levels require daily OHLC, so we use 1d data to calculate pivots
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate typical Camarilla levels from previous day's OHLC
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We align these to 6h timeframe
    def calculate_camarilla(high, low, close):
        if len(high) < 1:
            return np.full_like(high, np.nan), np.full_like(high, np.nan), \
                   np.full_like(high, np.nan), np.full_like(high, np.nan)
        # Daily range
        daily_range = high - low
        close_val = close
        # Camarilla levels
        r4 = close_val + daily_range * 1.1 / 2
        r3 = close_val + daily_range * 1.1 / 4
        s3 = close_val - daily_range * 1.1 / 4
        s4 = close_val - daily_range * 1.1 / 2
        return r4, r3, s3, s4
    
    r4_1d, r3_1d, s3_1d, s4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(50, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or 
            np.isnan(s4_1d_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Simple ATR-based stoploss: exit if price moves 2.5*ATR against position
            if position_side > 0:  # Long
                if price < entry_price - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if price > entry_price + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average) for confirmation
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Regime filter: 12h ADX > 25 = trending, < 25 = ranging
            is_trending = adx_12h_aligned[i] > 25
            
            if is_trending:
                # Trending market: breakout continuation at R4/S4
                # Long: price breaks above R4 with volume
                if price > r4_1d_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    signals[i] = SIZE
                # Short: price breaks below S4 with volume
                elif price < s4_1d_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:
                # Ranging market: mean reversion at extreme R3/S3 levels
                # Long: price rejects S3 (mean reversion up)
                if price < s3_1d_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    signals[i] = SIZE
                # Short: price rejects R3 (mean reversion down)
                elif price > r3_1d_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals