#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h price closing outside Bollinger Bands (20,2) with 1d trend filter (EMA34) and volume confirmation (>1.5x 20-period average).
# Long when close > upper band in uptrend (price > 1d EMA34), short when close < lower band in downtrend.
# Bollinger breakouts capture momentum bursts; volume filter avoids low-conviction moves.
# Trend filter ensures alignment with higher timeframe direction. Target: ~25-35 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Pre-compute Bollinger Bands (20,2) on 4h close
    close_series = prices['close']
    basis = close_series.rolling(window=20, min_periods=20).mean()
    dev = close_series.rolling(window=20, min_periods=20).std()
    upper = basis + 2 * dev
    lower = basis - 2 * dev
    upper = upper.values
    lower = lower.values
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Trend filter: price vs 1d EMA34
        uptrend = price > ema_34_1d_aligned[i]
        downtrend = price < ema_34_1d_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # Long: price closes above upper band in uptrend
                if uptrend and price > upper[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price closes below lower band in downtrend
                elif downtrend and price < lower[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions: close crosses back inside bands
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price closes below middle band (mean reversion)
                if price < basis.iloc[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price closes above middle band
                if price > basis.iloc[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_BollingerBreakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0