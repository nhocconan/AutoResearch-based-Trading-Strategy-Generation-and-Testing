#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h breakout of 4h Bollinger Bands with volume confirmation and 1d EMA50 trend filter
    # Bollinger Bands provide dynamic support/resistance. Breakouts with volume confirm
    # institutional participation. 1d EMA50 ensures alignment with higher timeframe trend.
    # This combination reduces false breakouts and improves win rate in both bull and bear markets.
    # Focus on 1h timeframe with strict entry conditions to limit trades to 15-37/year.
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for Bollinger Bands
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma_4h = pd.Series(close_4h).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_4h = pd.Series(close_4h).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_4h + (std_4h * bb_std)
    lower_bb = sma_4h - (std_4h * bb_std)
    
    # Align Bollinger Bands to 1h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_4h, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_4h, lower_bb)
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after EMA warmup
        # Skip if data not ready or outside session
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above upper BB with volume + price above 1d EMA50 (uptrend)
            if close[i] > upper_bb_aligned[i] and vol_spike[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: Breakdown below lower BB with volume + price below 1d EMA50 (downtrend)
            elif close[i] < lower_bb_aligned[i] and vol_spike[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: Price returns to middle BB or trend reversal vs 1d EMA50
            middle_bb = sma_4h  # Middle Bollinger Band
            middle_bb_aligned = align_htf_to_ltf(prices, df_4h, middle_bb)
            if position == 1:
                if close[i] < middle_bb_aligned[i] or close[i] < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if close[i] > middle_bb_aligned[i] or close[i] > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_Bollinger_Breakout_4hBB_1dEMA50_Volume_Session_v1"
timeframe = "1h"
leverage = 1.0