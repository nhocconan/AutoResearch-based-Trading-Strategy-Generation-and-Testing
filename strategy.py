#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width regime + 1d Donchian(20) breakout with volume confirmation
# Bollinger Band Width < 0.05 identifies low volatility squeeze (regime filter)
# 1d Donchian(20) breakout provides directional bias from higher timeframe structure
# Volume confirmation (>1.8x 20-period EMA) ensures institutional participation
# Designed for 6h timeframe targeting 12-37 trades/year (50-150 total over 4 years)
# Works in ranging markets (buy low volatility breakouts) and trending markets (continuation)
# Uses discrete position sizing (0.25) to balance return potential with drawdown control
# Avoids overtrading by requiring low volatility regime + HTF breakout + volume spike

name = "6h_BBW_Squeeze_1dDonchian20_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Donchian breakout
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d Donchian(20) channels
    donchian_high = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # 6h Bollinger Band Width (20, 2.0)
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = ma_20 + (2.0 * std_20)
    lower_band = ma_20 - (2.0 * std_20)
    bb_width = (upper_band - lower_band) / ma_20
    
    # Volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.8 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(bb_width[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Low volatility regime: Bollinger Band Width < 0.05 (squeeze condition)
        low_volatility_regime = bb_width[i] < 0.05
        
        # Breakout conditions from 1d Donchian channels
        bullish_breakout = close[i] > donchian_high_aligned[i]
        bearish_breakout = close[i] < donchian_low_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if low_volatility_regime and bullish_breakout and volume_confirmation[i]:
                # Long: Low volatility squeeze + upward breakout + volume confirmation
                signals[i] = 0.25
                position = 1
            elif low_volatility_regime and bearish_breakout and volume_confirmation[i]:
                # Short: Low volatility squeeze + downward breakout + volume confirmation
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Breakdown below 1d Donchian Low OR volatility expands significantly
            if close[i] < donchian_low_aligned[i] or bb_width[i] > 0.15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Breakout above 1d Donchian High OR volatility expands significantly
            if close[i] > donchian_high_aligned[i] or bb_width[i] > 0.15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals