#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX regime filter + volume confirmation
# Elder Ray measures bull/bear power via EMA13; ADX>25 filters trending markets; volume confirms momentum.
# Works in bull (buy on bull power expansion) and bear (sell on bear power expansion) regimes.
# Target: 12-25 trades/year (50-100 total over 4 years) to minimize fee drag.

name = "6h_ElderRay_1dADX_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate 1d ADX(14) for regime filter
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift()).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift()).abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate ATR(14) for 6h timeframe (for stoploss and volatility filter)
    tr1_6h = high[1:] - low[1:]
    tr2_6h = np.abs(high[1:] - close[:-1])
    tr3_6h = np.abs(low[1:] - close[:-1])
    tr_6h = np.concatenate([[np.nan], np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))])
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Bull Power and Bear Power (Elder Ray)
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    bull_power = high - ema_13_1d_aligned
    bear_power = low - ema_13_1d_aligned
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = max(50, 20, 14)  # warmup for EMA13, ADX, volume, ATR
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_13_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_adx = adx_1d_aligned[i]
        curr_atr = atr_6h[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits and stops
        if position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry
            stop_price = entry_price - 2.0 * atr_at_entry
            
            # Exit conditions:
            # 1. Stoploss hit
            # 2. Bear power expands (market weakening)
            # 3. ADX drops below 20 (trend ending)
            # 4. Volume confirmation lost
            if (curr_low <= stop_price or
                curr_bear_power < 0 and curr_bear_power < bear_power[i-1] or  # expanding bear power
                curr_adx < 20 or
                not curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry
            stop_price = entry_price + 2.0 * atr_at_entry
            
            # Exit conditions:
            # 1. Stoploss hit
            # 2. Bull power expands (market strengthening)
            # 3. ADX drops below 20 (trend ending)
            # 4. Volume confirmation lost
            if (curr_high >= stop_price or
                curr_bull_power > 0 and curr_bull_power > bull_power[i-1] or  # expanding bull power
                curr_adx < 20 or
                not curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Only trade when ADX > 25 (trending market)
            if curr_adx <= 25:
                signals[i] = 0.0
                continue
                
            # Require volume confirmation
            if not curr_volume_confirm:
                signals[i] = 0.0
                continue
            
            # Long entry: bull power expands AND bull power > 0 (strong buying pressure)
            if curr_bull_power > 0 and curr_bull_power > bull_power[i-1]:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
            # Short entry: bear power expands AND bear power < 0 (strong selling pressure)
            elif curr_bear_power < 0 and curr_bear_power < bear_power[i-1]:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals