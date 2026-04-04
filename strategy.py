#!/usr/bin/env python3
"""
Experiment #5272: 12h Donchian Breakout + Volume Spike + Chop Regime Filter
HYPOTHESIS: On 12h timeframe, Donchian(20) breakouts capture strong momentum moves. 
Volume confirmation (2x average volume) ensures breakout validity, while Choppiness Index (CHOP > 61.8) 
filters for ranging regimes where breakouts fail. In trending regimes (CHOP <= 61.8), we trade breakouts 
in direction of the trend. Uses discrete position sizing (0.25) to balance profit with drawdown control. 
Designed for 12-37 trades/year on 12h timeframe (50-150 total over 4 years) to minimize fee drag. 
Works in bull markets by catching upside breakouts and in bear markets by catching downside breakouts, 
while avoiding false breakouts in ranging conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5272_12h_donchian_breakout_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for trend filter (EMA50) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        ema_50 = pd.Series(df_1d['close']).ewm(span=50, min_periods=50, adjust=False).mean().shift(1).values
        ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    else:
        ema_50_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for Chop regime filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 14:
        # True Range
        tr1 = pd.Series(df_1w['high']).diff().abs()
        tr2 = (pd.Series(df_1w['high']) - pd.Series(df_1w['close']).shift(1)).abs()
        tr3 = (pd.Series(df_1w['low']) - pd.Series(df_1w['close']).shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # Directional Movement
        dm_plus = pd.Series(df_1w['high']).diff()
        dm_minus = -pd.Series(df_1w['low']).diff()
        dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
        dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
        # Smoothed TR, DM+ , DM- (Wilder's smoothing = EMA with alpha=1/period)
        atr_14 = pd.Series(tr).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
        dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
        dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
        # DI+ and DI-
        di_plus = 100 * dm_plus_14 / (atr_14 + 1e-10)
        di_minus = 100 * dm_minus_14 / (atr_14 + 1e-10)
        # DX and Chop
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        chop = 100 * np.log10(tr.rolling(window=14, min_periods=14).sum() / 
                            (np.sqrt(14) * np.abs(tr.rolling(window=14, min_periods=14).mean()) + 1e-10)) / np.log10(14)
        chop_aligned = align_htf_to_ltf(prices, df_1w, chop.values)
    else:
        chop_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Donchian Channels (20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 12h Indicators: Volume Spike (2x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, Chop warmup
    
    for i in range(warmup, n):
        # --- Session Filter: 00-24 UTC (12h timeframe, less restrictive) ---
        # 12h candles already filter to specific sessions, so we can use full day
        
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position when Donchian reverses or chop regime changes ---
        if in_position:
            # Check for Donchian reversal (price crosses opposite channel)
            if position_side > 0:  # Long position
                if price < donch_low[i]:  # Exit long if price breaks below lower channel
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                if price > donch_high[i]:  # Exit short if price breaks above upper channel
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout conditions
        breakout_up = price > donch_high[i]
        breakout_down = price < donch_low[i]
        
        # Volume confirmation
        vol_confirm = vol_spike[i]
        
        # Regime filter from 1w Chop (CHOP <= 61.8 = trending)
        trending_regime = chop_aligned[i] <= 61.8
        
        # Entry conditions: Breakout + volume confirmation + trending regime
        if breakout_up and vol_confirm and trending_regime:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif breakout_down and vol_confirm and trending_regime:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals