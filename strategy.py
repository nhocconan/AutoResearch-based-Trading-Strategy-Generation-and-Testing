#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Fractal breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA34 trend direction and Williams Fractals (bullish/bearish).
- Williams Fractals: Bearish fractal (sell signal) when high[n-2] < high[n] and high[n-1] < high[n] and high[n+1] < high[n] and high[n+2] < high[n].
                     Bullish fractal (buy signal) when low[n-2] > low[n] and low[n-1] > low[n] and low[n+1] > low[n] and low[n+2] > low[n].
- Trend Filter: 1d EMA34 must align with breakout direction (long: close > EMA34, short: close < EMA34).
- Volume Filter: Current 4h volume > 1.8 * 30-period average 4h volume to confirm strong momentum.
- Entry: Long when bullish fractal confirmed AND close > 1d EMA34 AND volume spike.
         Short when bearish fractal confirmed AND close < 1d EMA34 AND volume spike.
- Exit: Opposite fractal break (long exits when bearish fractal confirmed, short exits when bullish fractal confirmed).
- Signal size: 0.25 discrete to minimize fee drag.
- Designed to capture reversal points aligned with daily trend while filtering chop/whipsaws.
- Works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Williams Fractals (requires 2 extra bars for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    # Align with additional_delay_bars=2 for fractal confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate 4h volume average for confirmation (30-period)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 30)  # Need 34 for EMA, 30 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        bearish_fract = bearish_fractal_aligned[i]
        bullish_fract = bullish_fractal_aligned[i]
        ema_34_level = ema_34_1d_aligned[i]
        
        # Volume spike: current volume > 1.8 * 30-period average volume
        volume_spike = curr_volume > 1.8 * vol_ma_30[i]
        
        # Fractal confirmation (non-zero values indicate confirmed fractal)
        bearish_confirmed = bearish_fract != 0
        bullish_confirmed = bullish_fract != 0
        
        # Trend alignment conditions
        above_ema = curr_close > ema_34_level
        below_ema = curr_close < ema_34_level
        
        # Exit conditions: opposite fractal confirmation
        if position != 0:
            # Exit long: bearish fractal confirmed
            if position == 1:
                if bearish_confirmed:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: bullish fractal confirmed
            elif position == -1:
                if bullish_confirmed:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Fractal breakout with trend and volume filters
        if position == 0:
            # Long: bullish fractal confirmed AND above EMA34 AND volume spike
            long_condition = bullish_confirmed and above_ema and volume_spike
            
            # Short: bearish fractal confirmed AND below EMA34 AND volume spike
            short_condition = bearish_confirmed and below_ema and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Williams_Fractal_Breakout_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0