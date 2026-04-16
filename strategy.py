#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA(21) pullback to 4h VWAP with 1d ADX(14) trend filter and volume confirmation.
# Long when price pulls back to 4h VWAP during 1d uptrend (ADX>25 & +DI>-DI) AND 1h EMA(21) slope up AND volume > 1.5x 20-period average.
# Short when price pulls back to 4h VWAP during 1d downtrend (ADX>25 & +DI<+DI) AND 1h EMA(21) slope down AND volume > 1.5x 20-period average.
# Exit on opposite 1h EMA(21) cross or ATR(14) stoploss (2*ATR from entry).
# Uses discrete position size 0.20. Designed to capture trend continuation moves with volume confirmation in trending markets.
# Works in both bull and bear markets by requiring 1d ADX trend filter and volume confirmation, avoiding counter-trend trades.
# Target: 60-150 total trades over 4 years (15-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h Indicators: EMA(21) for entry timing ===
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_prev = np.roll(ema_21, 1)
    ema_21_prev[0] = np.nan
    ema_21_up = ema_21 > ema_21_prev
    ema_21_down = ema_21 < ema_21_prev
    
    # === 4h Indicators: VWAP for pullback zone ===
    df_4h = get_htf_data(prices, '4h')
    typical_price_4h = (df_4h['high'].values + df_4h['low'].values + df_4h['close'].values) / 3.0
    vol_4h = df_4h['volume'].values
    vwap_4h = (np.cumsum(typical_price_4h * vol_4h) / np.cumsum(vol_4h)).values
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    
    # === 1d Indicators: ADX(14) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_1d_smooth = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_1d_smooth
    minus_di = 100 * minus_dm_smooth / atr_1d_smooth
    
    # ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    dx = np.where(np.isnan(dx) | np.isinf(dx), 0, dx)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    # Trend conditions
    adx_strong = adx_1d_aligned > 25
    plus_di_gt_minus = plus_di_aligned > minus_di_aligned
    minus_di_gt_plus = minus_di_aligned > plus_di_aligned
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 1h ATR for stoploss ===
    tr1_h = pd.Series(high).diff()
    tr2_h = pd.Series(low).diff().abs()
    tr3_h = pd.Series(close).shift(1).diff().abs()
    tr_1h = pd.concat([tr1_h, tr2_h, tr3_h], axis=1).max(axis=1)
    atr_1h_raw = pd.Series(tr_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ADX/ATR/EMA)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_21[i]) or np.isnan(vwap_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_1h_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_1h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below 1h EMA(21)
            if price < ema_21[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above 1h EMA(21)
            if price > ema_21[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price near 4h VWAP (±0.5%) AND 1d uptrend AND EMA(21) slope up AND volume spike
            vwap_distance = abs(price - vwap_4h_aligned[i]) / vwap_4h_aligned[i]
            if (vwap_distance < 0.005 and adx_strong[i] and plus_di_gt_minus[i] and 
                ema_21_up[i] and vol_spike):
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: Price near 4h VWAP (±0.5%) AND 1d downtrend AND EMA(21) slope down AND volume spike
            elif (vwap_distance < 0.005 and adx_strong[i] and minus_di_gt_plus[i] and 
                  ema_21_down[i] and vol_spike):
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_EMA21_4hVWAP_1dADX_VolumeSpike_V1"
timeframe = "1h"
leverage = 1.0