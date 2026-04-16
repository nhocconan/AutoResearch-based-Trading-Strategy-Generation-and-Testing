#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d ADX trend strength with 4h RSI mean reversion and volume confirmation.
# Long when 1d ADX > 25 (strong trend) AND 4h RSI < 30 (oversold pullback) with volume > 1.5x median.
# Short when 1d ADX > 25 (strong trend) AND 4h RSI > 70 (overbought rally) with volume > 1.5x median.
# Uses discrete position size 0.25. Exits when RSI returns to neutral zone (40-60) or ATR stoploss hits (2.0x ATR).
# ADX filters for trending markets only, avoiding whipsaws in ranging conditions. RSI captures pullbacks within the trend.
# Volume confirmation ensures institutional participation. 4h timeframe targets ~25-40 trades/year (100-160 total) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 1d Indicators: ADX (14-period) ===
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(high_1d - np.roll(high_1d, 1))
    down_move = pd.Series(np.roll(low_1d, 1) - low_1d)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # === 4h Indicators: RSI (14-period) ===
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # === 4h Indicators: Volume Median (20-period) ===
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # === 4h Indicators: ATR (14-period) for stoploss ===
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to primary timeframe (4h)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)  # Wait for 1d close
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    # RSI and ATR are already on primary timeframe
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20)  # ADX needs 30 bars (14+14+2 for smoothing), volume median needs 20
    
    # Track position state and entry price for ATR stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(rsi[i]) or
            np.isnan(vol_median_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        price = close[i]
        adx_val = adx_1d_aligned[i]
        rsi_val = rsi[i]
        vol_median = vol_median_aligned[i]
        atr = atr_14[i]
        
        # Volume filter: current volume > 1.5x median volume
        volume_filter = volume[i] > (vol_median * 1.5)
        
        # Trend filter: 1d ADX > 25 indicates strong trend
        strong_trend = adx_val > 25
        
        # RSI conditions for mean reversion within trend
        rsi_oversold = rsi_val < 30
        rsi_overbought = rsi_val > 70
        rsi_neutral = (rsi_val >= 40) & (rsi_val <= 60)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when RSI returns to neutral (40-60) or ATR stoploss hit (2.0 * ATR below entry)
            if rsi_neutral or price <= entry_price - 2.0 * atr:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when RSI returns to neutral (40-60) or ATR stoploss hit (2.0 * ATR above entry)
            if rsi_neutral or price >= entry_price + 2.0 * atr:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Strong trend + oversold pullback + volume confirmation
            if strong_trend and rsi_oversold and volume_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Strong trend + overbought rally + volume confirmation
            elif strong_trend and rsi_overbought and volume_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_ADXTrend_RSIPullback_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0