#!/usr/bin/env python3
name = "12h_Williams_Fractal_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for Williams Fractal and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Williams Fractals on daily
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Fractals need 2 extra daily bars for confirmation (center bar + 2 after)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Daily EMA for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection on 12h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # enough for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish fractal + price above daily EMA34 + volume spike
            if bullish_fractal_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume[i] > vol_ma_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # Short: bearish fractal + price below daily EMA34 + volume spike
            elif bearish_fractal_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume[i] > vol_ma_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below daily EMA34 or fractal signal gone
            if close[i] < ema_34_1d_aligned[i] or not bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above daily EMA34 or fractal signal gone
            if close[i] > ema_34_1d_aligned[i] or not bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Williams Fractal on daily with 12h execution
# - Williams Fractals identify potential reversal points (bullish: low with two higher lows on each side; bearish: high with two lower highs)
# - Requires confirmation of 2 additional daily bars after the fractal forms (non-look-ahead)
# - Trade in direction of fractal only when price is above/below daily EMA34 (trend filter)
# - Volume spike (1.5x 20-period MA) confirms conviction
# - Works in both bull and bear markets: fractals work at turning points, trend filter avoids counter-trend trades
# - Exit when price crosses back below/above daily EMA34 or fractal signal invalidates
# - Position size 0.25 balances return and risk, targeting ~20-50 trades/year
# - Uses 12h timeframe to reduce noise vs lower timeframes while capturing multi-day swings
# - Fractal + EMA + volume combo is under-explored, avoids saturated strategies like pure Donchian/Camarilla breakouts