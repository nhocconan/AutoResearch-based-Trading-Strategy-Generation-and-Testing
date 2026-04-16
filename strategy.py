#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h EMA trend filter and volume confirmation
# Long when price > R3 AND 12h EMA50 > EMA200 AND 12h volume > 2.0x 20-period volume SMA
# Short when price < S3 AND 12h EMA50 < EMA200 AND 12h volume > 2.0x 20-period volume SMA
# Exit on price returning to pivot point (PP) or ATR stoploss (1.5 ATR)
# Uses proven Camarilla structure, trend filter for direction, volume for conviction
# Discrete sizing 0.25 targets 20-30 trades/year to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data once before loop for trend and volume filters
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    
    # === 12h Indicators: EMA50 and EMA200 for trend filter ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_200_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # === 12h Indicator: Volume SMA (20-period) for confirmation ===
    volume_12h = df_12h['volume'].values
    vol_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_sma_20_12h)
    
    # === 4h Indicator: Camarilla Pivot Levels from previous day ===
    # Use daily OHLC from previous completed day to calculate today's Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1 / 2
    # S3 = PP - (H - L) * 1.1 / 2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r3_1d = pp_1d + (high_1d - low_1d) * 1.1 / 2.0
    s3_1d = pp_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align daily Camarilla levels to 4h timeframe (completed bar delay)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 200  # Need 200 for EMA200 and daily data alignment
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0  # force flat outside session
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(vol_sma_20_12h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h volume (aligned)
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        if np.isnan(vol_12h_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 2.0x 20-period 12h volume SMA
        vol_threshold = vol_sma_20_12h_aligned[i] * 2.0
        vol_confirm = vol_12h_aligned[i] > vol_threshold
        
        # Trend filter: EMA50 > EMA200 for bullish, EMA50 < EMA200 for bearish
        bullish_trend = ema_50_aligned[i] > ema_200_aligned[i]
        bearish_trend = ema_50_aligned[i] < ema_200_aligned[i]
        
        # Price levels
        price = close[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        pp = pp_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit on price returning to pivot point or ATR stoploss
            if price <= pp or price <= entry_price - 1.5 * atr_14[i]:
                exit_signal = True
        elif position == -1:  # short position
            # Exit on price returning to pivot point or ATR stoploss
            if price >= pp or price >= entry_price + 1.5 * atr_14[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Price > R3 AND bullish trend AND volume confirmation
            if price > r3 and bullish_trend and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price < S3 AND bearish trend AND volume confirmation
            elif price < s3 and bearish_trend and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = 0.0  # maintain position
    
    return signals

name = "4h_CamarillaR3S3_12hEMATrend_Volume2.0x_v1"
timeframe = "4h"
leverage = 1.0