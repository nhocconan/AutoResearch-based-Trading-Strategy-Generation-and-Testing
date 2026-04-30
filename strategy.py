#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w/1d multi-timeframe confluence for regime detection.
# Uses 1w ADX(14) to identify strong trends (ADX > 25) and 1d RSI(14) for mean reversion in ranging markets.
# In strong trends (1w ADX > 25): trade 6h Donchian(20) breakouts in trend direction with volume confirmation (>1.5x avg).
# In ranging markets (1w ADX <= 25): fade 1d Bollinger Band(20,2) extremes with 6h RSI(14) confirmation.
# Uses ATR-based trailing stop (2.0x ATR) for risk management.
# Designed for low trade frequency (~12-30/year on 6h) to minimize fee drag while adapting to market regime.
# Works in bull/bear via trend following and in ranging markets via mean reversion at key levels.

name = "6h_1wADX_1dRSIBB_RegimeAdaptive_DonchianBreakout_BBFade_v1"
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
    
    # Load 1w data ONCE before loop for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for RSI/BB mean reversion signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1w ADX(14) for trend strength regime
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr_1w).ewm(span=14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (tr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # Align 1w ADX to 6h timeframe (wait for 1w bar to close)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate 1d RSI(14) for mean reversion
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain = np.concatenate([[0], gain])
    loss = np.concatenate([[0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align 1d RSI to 6h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 1d Bollinger Bands(20,2) for mean reversion levels
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Align 1d BB to 6h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Calculate 6h Donchian(20) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h RSI(14) for mean reversion confirmation
    delta_6h = np.diff(close)
    gain_6h = np.where(delta_6h > 0, delta_6h, 0)
    loss_6h = np.where(delta_6h < 0, -delta_6h, 0)
    gain_6h = np.concatenate([[0], gain_6h])
    loss_6h = np.concatenate([[0], loss_6h])
    avg_gain_6h = pd.Series(gain_6h).ewm(span=14, adjust=False).mean().values
    avg_loss_6h = pd.Series(loss_6h).ewm(span=14, adjust=False).mean().values
    rs_6h = avg_gain_6h / (avg_loss_6h + 1e-10)
    rsi_6h = 100 - (100 / (1 + rs_6h))
    
    # Calculate 6h ATR(14) for dynamic trailing stop
    tr1_6h = high[1:] - low[1:]
    tr2_6h = np.abs(high[1:] - close[:-1])
    tr3_6h = np.abs(low[1:] - close[:-1])
    tr_6h = np.concatenate([[np.max([tr1_6h[0], tr2_6h[0], tr3_6h[0]])], np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))])
    atr = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Regime filter: 1w ADX > 25 = trending, <= 25 = ranging
        is_trending = adx_aligned[i] > 25
        is_ranging = adx_aligned[i] <= 25
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_rsi_6h = rsi_6h[i]
        curr_rsi_1d = rsi_aligned[i]
        curr_upper_bb = upper_bb_aligned[i]
        curr_lower_bb = lower_bb_aligned[i]
        curr_highest_20 = highest_20[i]
        curr_lowest_20 = lowest_20[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            if is_trending:
                # Trend following: Donchian breakout with volume confirmation
                if curr_close > curr_highest_20 and curr_volume_spike:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                elif curr_close < curr_lowest_20 and curr_volume_spike:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    lowest_since_entry = curr_close
            else:  # ranging market
                # Mean reversion: fade Bollinger Band extremes with RSI confirmation
                if curr_close < curr_lower_bb and curr_rsi_1d < 30 and curr_rsi_6h < 35:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                elif curr_close > curr_upper_bb and curr_rsi_1d > 70 and curr_rsi_6h > 65:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_high > highest_since_entry:
                highest_since_entry = curr_high
            
            # Trailing stop: 2.0 * ATR below highest since entry
            if curr_close < highest_since_entry - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            
            # Trailing stop: 2.0 * ATR above lowest since entry
            if curr_close > lowest_since_entry + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals