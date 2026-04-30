#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with 1d EMA50 trend filter and volume confirmation.
# Uses 1d EMA50 for stable trend direction (more responsive than 1w for 12h timeframe).
# Requires volume > 2.0x 20-period average to confirm breakout strength.
# Only takes bullish fractal breaks above resistance in uptrend or bearish fractal breaks below support in downtrend.
# Added ATR-based stoploss (2.0x ATR) and profit target at opposite fractal level.
# Designed for very low trade frequency (~8-15 trades/year) to minimize fee drag and avoid overtrading.
# Williams Fractals provide reliable swing points that work in both trending and ranging markets.

name = "12h_WilliamsFractal_Breakout_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR for stoploss (using 14-period ATR on 12h)
    if n >= 14:
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    else:
        atr = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        
        # Calculate Williams Fractals on 1d timeframe (more reliable than lower TF)
        if len(df_1d) >= 5:
            # Williams Fractals: 5-bar pattern (high/low surrounded by 2 lower highs/higher lows)
            high_vals = df_1d['high'].values
            low_vals = df_1d['low'].values
            
            bullish_fractal = np.full(len(df_1d), np.nan)
            bearish_fractal = np.full(len(df_1d), np.nan)
            
            # Bullish fractal: lowest low in middle with 2 higher lows on each side
            for j in range(2, len(low_vals) - 2):
                if (low_vals[j] < low_vals[j-1] and low_vals[j] < low_vals[j-2] and
                    low_vals[j] < low_vals[j+1] and low_vals[j] < low_vals[j+2]):
                    bullish_fractal[j] = low_vals[j]
            
            # Bearish fractal: highest high in middle with 2 lower highs on each side
            for j in range(2, len(high_vals) - 2):
                if (high_vals[j] > high_vals[j-1] and high_vals[j] > high_vals[j-2] and
                    high_vals[j] > high_vals[j+1] and high_vals[j] > high_vals[j+2]):
                    bearish_fractal[j] = high_vals[j]
            
            # Align fractals to 12h timeframe with extra delay (fractals need confirmation)
            bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
            bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
        else:
            bullish_fractal_aligned = np.full(n, np.nan)
            bearish_fractal_aligned = np.full(n, np.nan)
        
        # Volume confirmation: volume > 2.0x 20-period average (strict to reduce trades)
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (2.0 * vol_ma_20)
        else:
            volume_confirm = False
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above bullish fractal (support), 1d EMA50 uptrend, volume spike
            if (not np.isnan(bullish_fractal_aligned[i]) and
                curr_close > bullish_fractal_aligned[i] and 
                curr_close > curr_ema_50_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below bearish fractal (resistance), 1d EMA50 downtrend, volume spike
            elif (not np.isnan(bearish_fractal_aligned[i]) and
                  curr_close < bearish_fractal_aligned[i] and 
                  curr_close < curr_ema_50_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: price breaks below bearish fractal (resistance), or ATR stoploss hit
            if (not np.isnan(bearish_fractal_aligned[i]) and curr_close < bearish_fractal_aligned[i]) or \
               curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price breaks above bullish fractal (support), or ATR stoploss hit
            if (not np.isnan(bullish_fractal_aligned[i]) and curr_close > bullish_fractal_aligned[i]) or \
               curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals