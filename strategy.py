#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d EMA34 trend and volume confirmation.
# Long when: price breaks above latest bullish Williams fractal (high) AND 1d close > 1d EMA34 AND 6h volume > 1.5x 20-period average
# Short when: price breaks below latest bearish Williams fractal (low) AND 1d close < 1d EMA34 AND 6h volume > 1.5x 20-period average
# Williams fractals identify significant swing points; breakouts with 1d trend and volume confirmation capture strong moves.
# Works in bull (breakouts with uptrend) and bear (breakdowns with downtrend) by trading with aligned 1d trend.
# Target: 12-30 trades/year on 6h. Discrete sizing 0.25 to minimize fee drag.

name = "6h_WilliamsFractal_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for Williams fractals and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Williams Fractals on 1d data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    high_vals = df_1d['high'].values
    low_vals = df_1d['low'].values
    n_1d = len(high_vals)
    
    bearish_fractal = np.full(n_1d, np.nan)
    bullish_fractal = np.full(n_1d, np.nan)
    
    for i in range(2, n_1d - 2):
        if (high_vals[i-2] < high_vals[i-1] and 
            high_vals[i] < high_vals[i-1] and
            high_vals[i-1] > high_vals[i-3] and
            high_vals[i-1] > high_vals[i+1]):
            bearish_fractal[i-1] = high_vals[i-1]  # center bar high
        
        if (low_vals[i-2] > low_vals[i-1] and 
            low_vals[i] > low_vals[i-1] and
            low_vals[i-1] < low_vals[i-3] and
            low_vals[i-1] < low_vals[i+1]):
            bullish_fractal[i-1] = low_vals[i-1]  # center bar low
    
    # Williams fractals need 2 extra 1d bars after center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for 1d EMA34
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_bearish_fractal = bearish_fractal_aligned[i]
        curr_bullish_fractal = bullish_fractal_aligned[i]
        curr_ema_34 = ema_34_1d_aligned[i]
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        # Calculate 6h volume MA on-the-fly using precomputed values would require HTF loading
        # Instead use rolling window on primary timeframe volume (acceptable for volatility filter)
        if i >= 20:
            vol_ma_6h = np.mean(volume[i-20:i])
            volume_confirm = curr_volume > (vol_ma_6h * 1.5)
        else:
            volume_confirm = False
        
        # 1d trend filter
        uptrend_1d = curr_close > curr_ema_34
        downtrend_1d = curr_close < curr_ema_34
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above latest bullish fractal AND 1d uptrend AND volume confirmation
            if (curr_high > curr_bullish_fractal and 
                uptrend_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: break below latest bearish fractal AND 1d downtrend AND volume confirmation
            elif (curr_low < curr_bearish_fractal and 
                  downtrend_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below latest bearish fractal (reversal) OR 1d trend turns down
            if (curr_close < curr_bearish_fractal or 
                not uptrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above latest bullish fractal (reversal) OR 1d trend turns up
            if (curr_close > curr_bullish_fractal or 
                not downtrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals