#!/usr/bin/env python3
"""
Experiment #5977: 4h Donchian(20) breakout + 1d/1w HTF bias + volume confirmation
HYPOTHESIS: Donchian breakouts on 4h aligned with higher timeframe bias (1d trend + 1w regime) 
capture sustained moves with lower noise. Daily EMA50 provides intermediate trend bias, 
weekly ADX regime filter avoids choppy markets. Volume >1.5x average confirms breakout strength. 
ATR trailing stop manages risk. Target 75-200 trades over 4 years.
Works in both bull/bear: HTF bias prevents counter-trend entries, volume confirmation 
avoids false breakouts, regime filter reduces whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5977_4h_donchian20_1d_1w_bias_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for EMA50 trend bias ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        ema_1d = pd.Series(df_1d['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    else:
        ema_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for ADX regime filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 14:
        # Calculate ADX(14) on weekly data
        whigh = df_1w['high'].values
        wlow = df_1w['low'].values
        wclose = df_1w['close'].values
        
        # True Range
        wtr1 = whigh - wlow
        wtr2 = np.abs(whigh - np.roll(wclose, 1))
        wtr3 = np.abs(wlow - np.roll(wclose, 1))
        wtr = np.maximum(wtr1, np.maximum(wtr2, wtr3))
        wtr[0] = wtr1[0]
        
        # Directional Movement
        wdm_plus = np.where((whigh - np.roll(whigh, 1)) > (np.roll(wlow, 1) - wlow), 
                           np.maximum(whigh - np.roll(whigh, 1), 0), 0)
        wdm_minus = np.where((np.roll(wlow, 1) - wlow) > (whigh - np.roll(whigh, 1)), 
                            np.maximum(np.roll(wlow, 1) - wlow, 0), 0)
        
        # Smoothed TR, DM+
        wtr_ma = pd.Series(wtr).ewm(span=14, min_periods=14, adjust=False).mean().values
        wdm_plus_ma = pd.Series(wdm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
        wdm_minus_ma = pd.Series(wdm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Directional Indicators
        wdi_plus = 100 * wdm_plus_ma / np.where(wtr_ma > 0, wtr_ma, 1)
        wdi_minus = 100 * wdm_minus_ma / np.where(wtr_ma > 0, wtr_ma, 1)
        
        # DX and ADX
        wdx = 100 * np.abs(wdi_plus - wdi_minus) / np.where((wdi_plus + wdi_minus) > 0, (wdi_plus + wdi_minus), 1)
        dadx = pd.Series(wdx).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Align to 4h timeframe
        dadx_aligned = align_htf_to_ltf(prices, df_1w, dadx)
    else:
        dadx_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 4h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 50, 14) + 1  # Donchian, volume avg, ATR, 1d EMA, 1w ADX + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(dadx_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below Donchian low (failed breakout)
                if price <= stop_price or price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above Donchian high (failed breakout)
                if price >= stop_price or price >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.5
        
        # HTF bias conditions:
        # 1d EMA50: price above/below intermediate trend
        above_1d_ema = price > ema_1d_aligned[i]
        below_1d_ema = price < ema_1d_aligned[i]
        
        # 1w ADX regime: only trade when trending (ADX > 25)
        trending_regime = dadx_aligned[i] > 25
        
        # Entry conditions: 
        # Long: breakout up with volume AND above 1d EMA AND trending regime
        # Short: breakout down with volume AND below 1d EMA AND trending regime
        long_setup = breakout_up and volume_confirmed and above_1d_ema and trending_regime
        short_setup = breakout_down and volume_confirmed and below_1d_ema and trending_regime
        
        if long_setup:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_setup:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals