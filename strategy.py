#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses Camarilla pivot levels from 4h timeframe for structure
# Breakout at H3/L3 with continuation when price closes beyond H4/L4
# 4h EMA50 as trend filter to avoid counter-trend trades
# Volume > 1.5x average confirms participation
# Session filter: 08-20 UTC to avoid low-liquidity hours
# Designed for ~20-40 trades/year to minimize fee drag while capturing strong moves
# Works in bull/bear via 4h EMA50 trend filter - only trades in direction of 4h trend

name = "1h_Camarilla_H3L3_4hEMA50_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivots and EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivot levels from 4h OHLC
    # Camarilla: H3 = close + 1.1*(high-low)*1.1/4, L3 = close - 1.1*(high-low)*1.1/4
    #          H4 = close + 1.1*(high-low)*1.1/2, L4 = close - 1.1*(high-low)*1.1/2
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate pivot range
    rng = high_4h - low_4h
    camarilla_h3 = close_4h + 1.1 * rng * 1.1 / 4
    camarilla_l3 = close_4h - 1.1 * rng * 1.1 / 4
    camarilla_h4 = close_4h + 1.1 * rng * 1.1 / 2
    camarilla_l4 = close_4h - 1.1 * rng * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4)
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = max(50, 14, 20)  # EMA, ATR, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema50_4h = ema_50_4h_aligned[i]
        curr_h3 = camarilla_h3_aligned[i]
        curr_l3 = camarilla_l3_aligned[i]
        curr_h4 = camarilla_h4_aligned[i]
        curr_l4 = camarilla_l4_aligned[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or price below L3 (reversal signal)
            if curr_close < entry_price - 2.5 * atr_at_entry or curr_close < curr_l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or price above H3 (reversal signal)
            if curr_close > entry_price + 2.5 * atr_at_entry or curr_close > curr_h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = curr_volume > 1.5 * curr_vol_ma
            
            # Long when price breaks above H3 with 4h EMA50 uptrend and volume confirmation
            # Requires close beyond H4 for confirmation of strong breakout
            if curr_high > curr_h3 and curr_close > curr_h4 and curr_close > curr_ema50_4h and vol_confirm:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
            # Short when price breaks below L3 with 4h EMA50 downtrend and volume confirmation
            # Requires close below L4 for confirmation of strong breakdown
            elif curr_low < curr_l3 and curr_close < curr_l4 and curr_close < curr_ema50_4h and vol_confirm:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals