#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R mean reversion with 1d EMA50 trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) AND 1d close > 1d EMA50 AND 12h volume > 1.5x 20-period average volume.
Short when Williams %R > -20 (overbought) AND 1d close < 1d EMA50 AND 12h volume > 1.5x 20-period average volume.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
Uses discrete position sizing (0.25) targeting ~15-35 trades/year on 12h timeframe.
Combines momentum oscillator mean reversion, trend filter, and volume confirmation for robustness across bull/bear regimes.
Williams %R calculated from prior 14 completed 12h bars, ensuring no look-ahead bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R(14) from prior completed 12h bars (no look-ahead)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_shifted = np.roll(high, 1)
    low_shifted = np.roll(low, 1)
    high_shifted[0] = np.nan
    low_shifted[0] = np.nan
    
    highest_high = pd.Series(high_shifted).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_shifted).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # 12h volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 50)  # Williams %R14 and EMA50 need 14 and 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr = williams_r[i]
        vol_ma_val = vol_ma[i]
        ema_val = ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: Oversold (WR < -80) AND bullish trend (1d close > EMA50) AND volume confirmation
            if wr < -80 and close[i] > ema_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Overbought (WR > -20) AND bearish trend (1d close < EMA50) AND volume confirmation
            elif wr > -20 and close[i] < ema_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Williams %R crosses above -50 (for longs) or below -50 (for shorts)
            if position == 1 and wr > -50:
                exit_signal = True
            elif position == -1 and wr < -50:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_MeanReversion_1dEMA50_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0