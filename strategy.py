#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA(34) trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 AND 1d EMA(34) trending up AND volume > 1.5x 20-period average.
# Short when price breaks below Camarilla S3 AND 1d EMA(34) trending down AND volume > 1.5x 20-period average.
# Exit on opposite Camarilla level (S3 for long, R3 for short) or ATR-based stoploss (2*ATR).
# Uses discrete position size 0.25. Designed to capture institutional breakouts with volume confirmation.
# Works in both bull and bear markets by requiring 1d trend filter and volume confirmation, avoiding false breakouts.
# Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Camarilla Pivot Levels (based on previous day) ===
    # Camarilla levels calculated from previous day's OHLC
    # R3 = close + 1.1*(high - low)*1.1/4
    # S3 = close - 1.1*(high - low)*1.1/4
    # We need to align daily OHLC to 4h bars
    
    df_1d = get_htf_data(prices, '1d')
    # Calculate Camarilla levels from 1d OHLC
    cam_r3_1d = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) * 1.1 / 4
    cam_s3_1d = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (each 1d value applies to 16 consecutive 4h bars)
    cam_r3_aligned = align_htf_to_ltf(prices, df_1d, cam_r3_1d.values)
    cam_s3_aligned = align_htf_to_ltf(prices, df_1d, cam_s3_1d.values)
    
    # === 1d Indicators: EMA(34) for trend ===
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_up = ema_34_aligned > np.roll(ema_34_aligned, 1)
    ema_down = ema_34_aligned < np.roll(ema_34_aligned, 1)
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 4h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h_raw = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 60 periods needed for EMA/ATR)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(cam_r3_aligned[i]) or np.isnan(cam_s3_aligned[i]) or np.isnan(ema_34_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_4h_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_4h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Camarilla S3
            if price < cam_s3_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Camarilla R3
            if price > cam_r3_aligned[i]:
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
            # LONG: Price breaks above Camarilla R3 AND EMA trending up AND volume spike
            if price > cam_r3_aligned[i] and ema_up[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S3 AND EMA trending down AND volume spike
            elif price < cam_s3_aligned[i] and ema_down[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0