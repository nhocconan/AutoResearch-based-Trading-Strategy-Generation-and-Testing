#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Bollinger Bands (20, 2) for volatility regime ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period SMA
    sma_20 = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= 19:
            sma_20[i] = np.mean(close_1d[i-19:i+1])
    
    # Calculate 20-period standard deviation
    std_20 = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= 19:
            std_20[i] = np.std(close_1d[i-19:i+1])
    
    # Upper and lower Bollinger Bands
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Bollinger Band Width (normalized)
    bb_width = np.where(sma_20 != 0, (upper_bb - lower_bb) / sma_20, 0)
    
    # === 1d ADX (14-period) for trend strength ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Wilder's smoothing for TR and DM
    atr_14 = np.full_like(tr, np.nan)
    plus_dm_14 = np.full_like(plus_dm, np.nan)
    minus_dm_14 = np.full_like(minus_dm, np.nan)
    
    if len(tr) >= 14:
        atr_14[13] = np.mean(tr[:14])
        plus_dm_14[13] = np.mean(plus_dm[:14])
        minus_dm_14[13] = np.mean(minus_dm[:14])
        
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
            plus_dm_14[i] = (plus_dm_14[i-1] * 13 + plus_dm[i]) / 14
            minus_dm_14[i] = (minus_dm_14[i-1] * 13 + minus_dm[i]) / 14
    
    # Directional Indicators
    plus_di = np.where(atr_14 != 0, 100 * plus_dm_14 / atr_14, 0)
    minus_di = np.where(atr_14 != 0, 100 * minus_dm_14 / atr_14, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = np.full_like(dx, np.nan)
    
    if len(dx) >= 14:
        adx[13] = np.mean(dx[:14])
        for i in range(14, len(dx)):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # === 1d RSI (14-period) for momentum ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    period = 14
    for i in range(len(gain)):
        if i < period:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i]) / i
                avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i]) / i
        else:
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[avg_loss == 0] = 100
    
    # === 6h Volume confirmation ===
    df_6h = get_htf_data(prices, '6h')
    volume_6h = df_6h['volume'].values
    
    # Calculate 20-period average volume on 6h timeframe
    vol_ma_20 = np.full_like(volume_6h, np.nan)
    for i in range(len(volume_6h)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_6h[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume_6h[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume_6h[0]
    
    # Volume confirmation: current 6h volume > 1.5x 20-period average
    vol_confirm = volume_6h > vol_ma_20 * 1.5
    
    # Align all indicators to 6h timeframe
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    vol_confirm_aligned = align_htf_to_ltf(prices, df_6h, vol_confirm)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_width_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_confirm_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Regime filter: Low volatility (BB width < 0.05) AND weak trend (ADX < 25)
        range_market = (bb_width_aligned[i] < 0.05) and (adx_aligned[i] < 25)
        
        # Entry logic: only enter when flat AND volume confirmation
        if position == 0:
            if range_market:
                # Mean reversion in ranging market
                # Long: RSI < 30 (oversold)
                if rsi_1d_aligned[i] < 30:
                    signals[i] = 0.25
                    position = 1
                    continue
                # Short: RSI > 70 (overbought)
                elif rsi_1d_aligned[i] > 70:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Exit logic
        elif position == 1:
            # Exit long: RSI crosses above 50 or regime changes
            if (rsi_1d_aligned[i] > 50) or not range_market:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses below 50 or regime changes
            if (rsi_1d_aligned[i] < 50) or not range_market:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BBWidth_ADX_RSI_MeanReversion_Volume"
timeframe = "6h"
leverage = 1.0