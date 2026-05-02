#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d ADX trend filter and volume confirmation
# Camarilla pivot levels provide precise intraday support/resistance based on prior day's range
# Breakouts above R3 or below S3 with 1d trend alignment (ADX > 25) capture strong momentum
# Volume confirmation (>1.5x 20-period average) filters weak breakouts
# Discrete sizing 0.25 targets 50-150 trades over 4 years (12-37/year)
# Works in bull/bear by only taking breakouts in direction of 1d trend

name = "12h_Camarilla_R3S3_Breakout_1dADX25_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Get 1d data for ADX and prior day's range
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff().copy()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)
    
    tr1 = pd.Series(df_1d['high']) - pd.Series(df_1d['low'])
    tr2 = abs(pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1))
    tr3 = abs(pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (abs(plus_di + minus_di))) * 100
    adx = dx.rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 12h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Get 1d DI values for trend direction
    plus_di_1d = 100 * (plus_dm.rolling(window=14, min_periods=14).mean() / atr)
    minus_di_1d = 100 * (minus_dm.rolling(window=14, min_periods=14).mean() / atr)
    plus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, plus_di_1d.values)
    minus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, minus_di_1d.values)
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Calculate Camarilla levels for each 12h bar using prior 1d bar's OHLC
    # We need to map each 12h bar to the most recent completed 1d bar
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    # For each 12h bar, find the prior completed 1d bar
    for i in range(n):
        current_time = open_time.iloc[i]
        # Find 1d bars that completed before this 12h bar
        completed_1d = df_1d[df_1d['open_time'] < current_time]
        if len(completed_1d) > 0:
            # Use the most recent completed 1d bar
            prior_day = completed_1d.iloc[-1]
            h, l, c = prior_day['high'], prior_day['low'], prior_day['close']
            camarilla_r3[i] = c + (h - l) * 1.1 / 4
            camarilla_s3[i] = c - (h - l) * 1.1 / 4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price > R3 with 1d uptrend (ADX > 25 and +DI > -DI)
            long_breakout = close[i] > camarilla_r3[i]
            # Short breakdown: price < S3 with 1d downtrend (ADX > 25 and -DI > +DI)
            short_breakout = close[i] < camarilla_s3[i]
            
            # 1d ADX trend filter: ADX > 25 indicates strong trend
            adx_strong = adx_aligned[i] > 25
            
            adx_long = adx_strong and (plus_di_1d_aligned[i] > minus_di_1d_aligned[i])
            adx_short = adx_strong and (minus_di_1d_aligned[i] > plus_di_1d_aligned[i])
            
            if long_breakout and adx_long and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif short_breakout and adx_short and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < S3 or ADX weakens (< 20)
            if close[i] < camarilla_s3[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > R3 or ADX weakens (< 20)
            if close[i] > camarilla_r3[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals