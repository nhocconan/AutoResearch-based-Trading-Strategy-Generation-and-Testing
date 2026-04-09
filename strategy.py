#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d ADX regime filter
# In strong trends (ADX > 25): Williams %R mean reversion (long when < -80, short when > -20)
# In weak trends/ranging (ADX <= 25): Williams %R trend following (long when crosses above -50, short when crosses below -50)
# Uses discrete position sizing 0.25 to target ~12-37 trades/year and minimize fee drag
# Works in bull/bear markets: adapts to regime, avoiding whipsaws in ranging markets and catching trends

name = "6h_1d_williamsr_adx_regime_v2"
timeframe = "6h"
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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for ADX
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Calculate 1d +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Calculate 1d smoothed +DM, -DM, and TR
    smoothed_plus_dm = wilders_smoothing(plus_dm, 14)
    smoothed_minus_dm = wilders_smoothing(minus_dm, 14)
    smoothed_tr = wilders_smoothing(tr, 14)
    
    # Calculate 1d +DI and -DI
    plus_di = np.where(smoothed_tr != 0, 100 * smoothed_plus_dm / smoothed_tr, 0)
    minus_di = np.where(smoothed_tr != 0, 100 * smoothed_minus_dm / smoothed_tr, 0)
    
    # Calculate 1d DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_1d = wilders_smoothing(dx, 14)
    
    # Calculate 6h Williams %R (14-period)
    def williams_r(high, low, close, period):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        wr = np.where((highest_high - lowest_low) != 0, 
                      -100 * (highest_high - close) / (highest_high - lowest_low), 
                      -50)
        return wr
    
    wr_6h = williams_r(high, low, close, 14)
    
    # Align 1d indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(wr_6h[i])):
            signals[i] = 0.0
            continue
        
        adx = adx_1d_aligned[i]
        wr = wr_6h[i]
        
        if position == 1:  # Long position
            if adx > 25:  # Strong trend: mean reversion
                # Exit long if Williams %R rises above -20 (overbought)
                if wr > -20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Weak trend/ranging: trend following
                # Exit long if Williams %R falls below -50
                if wr < -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            if adx > 25:  # Strong trend: mean reversion
                # Exit short if Williams %R falls below -80 (oversold)
                if wr < -80:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Weak trend/ranging: trend following
                # Exit short if Williams %R rises above -50
                if wr > -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if adx > 25:  # Strong trend: mean reversion
                # Enter long if Williams %R < -80 (oversold)
                if wr < -80:
                    position = 1
                    signals[i] = 0.25
                # Enter short if Williams %R > -20 (overbought)
                elif wr > -20:
                    position = -1
                    signals[i] = -0.25
            else:  # Weak trend/ranging: trend following
                # Enter long if Williams %R crosses above -50
                if wr > -50 and (i == 100 or wr_6h[i-1] <= -50):
                    position = 1
                    signals[i] = 0.25
                # Enter short if Williams %R crosses below -50
                elif wr < -50 and (i == 100 or wr_6h[i-1] >= -50):
                    position = -1
                    signals[i] = -0.25
    
    return signals