#!/usr/bin/env python3
"""
Experiment #266: 4h Donchian(20) Breakout + 1d EMA Trend + Volume Spike + ATR Stoploss

HYPOTHESIS: Donchian channel breakouts capture strong momentum moves. When aligned with 1d EMA trend and confirmed by volume spikes, these breakouts have higher follow-through. The 1d EMA provides the primary trend filter, avoiding counter-trend breakouts that fail in ranging/weak markets. Volume spike confirms institutional participation. ATR-based stoploss manages risk. Targets 25-50 trades/year on 4h timeframe (100-200 total over 4 years) to minimize fee drag while capturing high-probability trend continuation moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_1d_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Donchian Channel(20) - upper and lower bands
    def donchian_channels(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    # Volume Spike: volume > 2.0 * 20-period average volume
    def volume_spike(vol, period):
        avg_vol = pd.Series(vol).rolling(window=period, min_periods=period).mean().values
        spike = vol > (2.0 * avg_vol)
        return spike
    
    # Calculate indicators
    donch_upper, donch_lower = donchian_channels(high, low, 20)
    vol_spike = volume_spike(volume, 20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donch_upper[i]) or 
            np.isnan(donch_lower[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price retouches Donchian middle (mean reversion)
                middle = (donch_upper[i] + donch_lower[i]) / 2.0
                if abs(close[i] - middle) < 0.1 * (donch_upper[i] - donch_lower[i]):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price retouches Donchian middle
                middle = (donch_upper[i] + donch_lower[i]) / 2.0
                if abs(close[i] - middle) < 0.1 * (donch_upper[i] - donch_lower[i]):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Breakout conditions
        breakout_up = close[i] > donch_upper[i]
        breakout_dn = close[i] < donch_lower[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Trend filter: 1d EMA
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Long: Bullish breakout with volume in uptrend
        if breakout_up and vol_confirmed and uptrend:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        # Short: Bearish breakout with volume in downtrend
        elif breakout_dn and vol_confirmed and downtrend:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals