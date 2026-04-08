# 4h_volatility_breakout_v5
# Hypothesis: Volatility breakout with 12h trend filter and volume confirmation.
# Uses ATR-based breakout channels with 12h trend filter to avoid counter-trend trades.
# Target: 25-35 trades/year to minimize fee drag while capturing volatility expansion.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_volatility_breakout_v5"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h trend filter - load once before loop
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h data for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 4h indicators
    # EMA20 for dynamic center line
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(14) for volatility bands
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Upper and lower bands (EMA20 ± 2.5*ATR)
    upper_band = ema20 + 2.5 * atr
    lower_band = ema20 - 2.5 * atr
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(ema20[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(avg_volume[i]) or np.isnan(ema50_12h_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # 12h trend filter
        trend_up = ema50_12h_aligned[i] > ema50_12h_aligned[i-1]
        trend_down = ema50_12h_aligned[i] < ema50_12h_aligned[i-1]
        
        if position == 1:  # Long position
            # Exit: price below EMA20 or volatility contraction
            if close[i] < ema20[i] or atr[i] < atr[i-1] * 0.85:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above EMA20 or volatility contraction
            if close[i] > ema20[i] or atr[i] < atr[i-1] * 0.85:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.4 * avg_volume[i]
            
            if volume_ok:
                # Long: breakout above upper band in uptrend
                if trend_up and close[i] > upper_band[i] and close[i-1] <= upper_band[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short: breakdown below lower band in downtrend
                elif trend_down and close[i] < lower_band[i] and close[i-1] >= lower_band[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals