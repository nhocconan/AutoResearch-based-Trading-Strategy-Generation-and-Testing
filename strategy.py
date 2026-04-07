#!/usr/bin/env python3
"""
6h_ema_rsi_trend_volume_v1
Hypothesis: On 6-hour timeframe, enter trades only when EMA(50) trend, RSI(2) extreme, and volume spike align.
Long when EMA(50) rising, RSI(2) < 10 (oversold), and volume > 2x 20-period average.
Short when EMA(50) falling, RSI(2) > 90 (overbought), and volume > 2x 20-period average.
Exit when RSI(2) crosses back to neutral (40-60 range).
Uses contrarian momentum within trend context to capture mean-reversion bounces in both bull and bear markets.
Low frequency design targets 15-25 trades/year to minimize fee impact.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ema_rsi_trend_volume_v1"
timeframe = "6h"
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
    
    # EMA(50) for trend filter
    ema_period = 50
    ema = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    # RSI(2) for entry signals
    rsi_period = 2
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(20, 50), n):
        # Skip if data not available
        if (np.isnan(ema[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 2.0 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 40 (exiting oversold)
            if rsi[i] >= 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 60 (exiting overbought)
            if rsi[i] <= 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation
            if vol_ok:
                # EMA trend direction
                ema_rising = ema[i] > ema[i-1]
                ema_falling = ema[i] < ema[i-1]
                
                # Long: EMA up + RSI extremely oversold
                if ema_rising and rsi[i] < 10:
                    position = 1
                    signals[i] = 0.25
                # Short: EMA down + RSI extremely overbought
                elif ema_falling and rsi[i] > 90:
                    position = -1
                    signals[i] = -0.25
    
    return signals