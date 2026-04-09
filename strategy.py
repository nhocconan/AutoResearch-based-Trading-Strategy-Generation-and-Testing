#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams Fractals with volume confirmation and ATR filter
# Williams Fractals identify major swing points that work in both bull and bear markets
# Fade at bearish fractal (sell at swing high, expect reversion)
# Fade at bullish fractal (buy at swing low, expect reversion)
# Volume confirmation (current 12h volume > 1.5x 20-period average) filters false signals
# ATR filter ensures sufficient volatility (avoid choppy low-vol periods)
# Position size fixed at 0.25 to minimize fee churn and control drawdown
# Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_williams_fractal_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams Fractals (5-bar: 2 left, center, 2 right)
    # Bearish fractal: high[n] > high[n-2], high[n] > high[n-1], high[n] > high[n+1], high[n] > high[n+2]
    # Bullish fractal: low[n] < low[n-2], low[n] < low[n-1], low[n] < low[n+1], low[n] < low[n+2]
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i-1] and 
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        if (low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i-1] and 
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Williams Fractals need 2 extra 1d bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 1d ATR (14-period) for volatility filtering
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1)) if 'close_1d' in locals() else np.abs(high_1d - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1)) if 'close_1d' in locals() else np.abs(low_1d - np.roll(df_1d['close'].values, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(atr_aligned[i]) or np.isnan(vol_ma_20[i]) or
            atr_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 12h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Volatility filter: only trade when ATR is above its 50-period average (avoid low-vol chop)
        atr_ma_50 = pd.Series(atr_aligned).rolling(window=50, min_periods=50).mean()
        if len(atr_ma_50) > i:
            vol_filter = atr_aligned[i] > atr_ma_50.iloc[i]
        else:
            vol_filter = True  # Not enough data for MA, allow trading
            
        if not vol_filter:
            signals[i] = 0.0
            continue
        
        # Fixed position size to minimize fee churn
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit on retracement to bullish fractal or stop at bearish fractal breakdown
            if close[i] < bullish_fractal_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] < bearish_fractal_aligned[i]:  # Stop loss at bearish fractal breakdown
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to bearish fractal or stop at bullish fractal breakout
            if close[i] > bearish_fractal_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] > bullish_fractal_aligned[i]:  # Stop loss at bullish fractal breakout
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Williams Fractal trading with volume and volatility confirmation
            # Fade at bearish fractal (sell at swing high, expect reversion)
            # Fade at bullish fractal (buy at swing low, expect reversion)
            if volume_confirmed:
                # Fade at bearish fractal (sell at resistance, expect reversion to support)
                if close[i] < bearish_fractal_aligned[i] and close[i] > bullish_fractal_aligned[i]:
                    position = -1
                    signals[i] = -position_size
                # Fade at bullish fractal (buy at support, expect reversion to resistance)
                elif close[i] > bullish_fractal_aligned[i] and close[i] < bearish_fractal_aligned[i]:
                    position = 1
                    signals[i] = position_size
    
    return signals