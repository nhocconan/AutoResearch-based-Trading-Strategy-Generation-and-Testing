#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike
# Uses Camarilla pivot levels from 4h timeframe for structural support/resistance.
# Only takes long breakouts above R1 in uptrend (price > 4h EMA50) and short breakdowns below S1 in downtrend.
# Volume confirmation (>2.0x 20-period average) filters weak breakouts.
# Session filter (08-20 UTC) reduces noise trades outside active market hours.
# Designed for ~20-40 trades/year on 1h timeframe to minimize fee drag while capturing high-probability moves.
# Works in both bull and bear markets via 4h trend filter - only trades breakouts in trend direction.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours for filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla pivots and EMA50 trend (HTF = 4h)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivot levels from 4h OHLC
    # Camarilla: R1 = close + 1.091*(high-low), S1 = close - 1.091*(high-low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_arr = df_4h['close'].values
    
    camarilla_r1 = close_4h_arr + 1.091 * (high_4h - low_4h)
    camarilla_s1 = close_4h_arr - 1.091 * (high_4h - low_4h)
    
    # Align Camarilla levels to 1h timeframe (with 1-bar delay for completed 4h bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Calculate 20-period average volume for confirmation (on 1h data)
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
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50_4h = ema_50_4h_aligned[i]
        curr_r1 = camarilla_r1_aligned[i]
        curr_s1 = camarilla_s1_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        curr_atr = atr[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or price closes below Camarilla S1 (mean reversion)
            if curr_close < entry_price - 2.0 * curr_atr or curr_close < curr_s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or price closes above Camarilla R1 (mean reversion)
            if curr_close > entry_price + 2.0 * curr_atr or curr_close > curr_r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Long entry: bullish breakout above R1 in uptrend (price > 4h EMA50)
            if vol_confirm and curr_close > curr_ema50_4h:
                if curr_high > curr_r1:  # Breakout above R1
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
            # Short entry: bearish breakdown below S1 in downtrend (price < 4h EMA50)
            elif vol_confirm and curr_close < curr_ema50_4h:
                if curr_low < curr_s1:  # Breakdown below S1
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals