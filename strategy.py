#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla H4/L4 breakout with 1w EMA50 trend filter and volume spike
# Uses Camarilla pivot levels from weekly timeframe for stronger structural support/resistance.
# Only takes long breakouts above H4 in uptrend (price > 1w EMA50) and short breakdowns below L4 in downtrend.
# Volume confirmation (>2.0x 20-period average) filters weak breakouts.
# Designed for ~25-50 trades/year on 4h timeframe to minimize fee drag while capturing high-probability moves.
# Works in both bull and bear markets via 1w trend filter - only trades breakouts in trend direction.

name = "4h_Camarilla_H4L4_Breakout_1wEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla pivots and EMA50 trend (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla pivot levels from 1w OHLC
    # Camarilla: H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    #            H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    
    camarilla_h4 = close_1w_arr + 1.5 * (high_1w - low_1w)
    camarilla_l4 = close_1w_arr - 1.5 * (high_1w - low_1w)
    
    # Align Camarilla levels to 4h timeframe (with 1-bar delay for completed 1w bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    # Calculate 20-period average volume for confirmation (on 4h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # Volume MA and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_h4 = camarilla_h4_aligned[i]
        curr_l4 = camarilla_l4_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        curr_atr = atr[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or price closes below Camarilla L4 (mean reversion)
            if curr_close < entry_price - 2.0 * curr_atr or curr_close < curr_l4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or price closes above Camarilla H4 (mean reversion)
            if curr_close > entry_price + 2.0 * curr_atr or curr_close > curr_h4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Long entry: bullish breakout above H4 in uptrend (price > 1w EMA50)
            if vol_confirm and curr_close > curr_ema50_1w:
                if curr_high > curr_h4:  # Breakout above H4
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
            # Short entry: bearish breakdown below L4 in downtrend (price < 1w EMA50)
            elif vol_confirm and curr_close < curr_ema50_1w:
                if curr_low < curr_l4:  # Breakdown below L4
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals