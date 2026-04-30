#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation
# Bull Power = High - EMA13(close); Bear Power = EMA13(close) - Low
# Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) + 1d ADX > 25 (trending) + volume > 1.5x 20-period average
# Short: Bear Power > 0 AND Bull Power < 0 (bearish momentum) + 1d ADX > 25 + volume confirmation
# Uses discrete sizing 0.25 to balance profit and fee drag. Target: 80-160 total trades over 4 years (20-40/year).
# Works in both bull and bear via 1d ADX filter - only trades when higher timeframe is trending.

name = "6h_ElderRay_1dADX_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    # Calculate 1d ADX(14) for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = df_1d['high'][1:] - df_1d['low'][1:]
    tr2 = np.abs(df_1d['high'][1:] - df_1d['close'][:-1])
    tr3 = np.abs(df_1d['low'][1:] - df_1d['close'][:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.concatenate([[np.nan], np.where((df_1d['high'][1:] - df_1d['high'][:-1]) > (df_1d['low'][:-1] - df_1d['low'][1:]), 
                                                 np.maximum(df_1d['high'][1:] - df_1d['high'][:-1], 0), 0)])
    dm_minus = np.concatenate([[np.nan], np.where((df_1d['low'][:-1] - df_1d['low'][1:]) > (df_1d['high'][1:] - df_1d['high'][:-1]), 
                                                  np.maximum(df_1d['low'][:-1] - df_1d['low'][1:], 0), 0)])
    
    # Smoothed TR, DM+, DM-
    tr_14 = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_14 = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 20, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_14_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_adx_14 = adx_14_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        curr_atr = atr_14[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume confirmation with Elder Ray signals and 1d ADX > 25
            if curr_volume_confirm and curr_adx_14 > 25:
                # Bullish: Bull Power > 0 AND Bear Power < 0 (momentum shifting up)
                if curr_bull_power > 0 and curr_bear_power < 0:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish: Bear Power > 0 AND Bull Power < 0 (momentum shifting down)
                elif curr_bear_power > 0 and curr_bull_power < 0:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry
            stop_loss = entry_price - 2.0 * curr_atr
            # Exit: Stoploss hit OR Elder Ray turns bearish OR loses 1d trend
            if curr_low <= stop_loss or (curr_bull_power <= 0 and curr_bear_power >= 0) or curr_adx_14 < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry
            stop_loss = entry_price + 2.0 * curr_atr
            # Exit: Stoploss hit OR Elder Ray turns bullish OR loses 1d trend
            if curr_high >= stop_loss or (curr_bear_power <= 0 and curr_bull_power >= 0) or curr_adx_14 < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals