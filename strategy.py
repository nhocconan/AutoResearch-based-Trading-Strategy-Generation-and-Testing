#!/usr/bin/env python3
# mtf_1h_rsi_mean_reversion_4h1d_v1
# Hypothesis: 1h RSI mean reversion with 4h/1d trend filter and volume confirmation.
# In bull/bear markets: RSI extremes (overbought/oversold) revert to mean, especially when aligned with higher timeframe trend.
# 4h EMA provides trend direction, 1d EMA filters extreme counter-trend moves, volume confirms reversal strength.
# Target: 15-37 trades/year (60-150 total over 4 years) with discrete sizing to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_mean_reversion_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h HTF data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # 1d HTF data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    close_1d = df_1d['close'].values
    
    # 4h EMA21 for trend direction
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d EMA50 for trend filter (stronger trend)
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1h RSI(14) for mean reversion signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral (50) OR trend turns bearish
            if rsi[i] >= 50 or close[i] < ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral (50) OR trend turns bullish
            if rsi[i] <= 50 or close[i] > ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long: RSI oversold (<30) + price above 4h EMA (uptrend) + price above 1d EMA (strong uptrend filter)
                if rsi[i] < 30 and close[i] > ema_4h_aligned[i] and close[i] > ema_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short: RSI overbought (>70) + price below 4h EMA (downtrend) + price below 1d EMA (strong downtrend filter)
                elif rsi[i] > 70 and close[i] < ema_4h_aligned[i] and close[i] < ema_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals