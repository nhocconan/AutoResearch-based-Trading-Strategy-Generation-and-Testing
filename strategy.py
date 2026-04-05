#!/usr/bin/env python3
"""
Experiment #7788: 12-hour Camarilla pivot reversal with weekly trend filter and volume confirmation.
Hypothesis: Price rejecting Camarilla H4/L4 levels on 12h with volume >1.5x 20-period MA and aligned weekly trend (price > weekly EMA20 for longs, < for shorts) captures reversals in both bull and bear markets. Weekly trend filter ensures alignment with higher timeframe momentum while avoiding counter-trend trades. Targets 75-150 trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7788_12h_camarilla_pivot_reversal_weekly_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # Use previous day for Camarilla calculation
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
WEEKLY_EMA = 20

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate weekly EMA for trend filter
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=WEEKLY_EMA, adjust=False, min_periods=WEEKLY_EMA).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Calculate daily data for Camarilla pivots
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate Camarilla levels (H4, L4) from previous day
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    camarilla_h4 = close_daily + 1.5 * (high_daily - low_daily)
    camarilla_l4 = close_daily - 1.5 * (high_daily - low_daily)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_l4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WEEKLY_EMA, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_weekly_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market regime from weekly trend
        bull_regime = close[i] > ema_weekly_aligned[i]   # price above weekly EMA
        bear_regime = close[i] < ema_weekly_aligned[i]   # price below weekly EMA
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Reversal conditions at Camarilla levels
        # Long: price rejects L4 (bounces off support) in bull regime
        long_rejection = (low[i] <= camarilla_l4_aligned[i] and close[i] > camarilla_l4_aligned[i]) and bull_regime
        # Short: price rejects H4 (gets rejected at resistance) in bear regime
        short_rejection = (high[i] >= camarilla_h4_aligned[i] and close[i] < camarilla_h4_aligned[i]) and bear_regime
        
        # Entry conditions with volume confirmation
        long_entry = long_rejection and volume_confirmed
        short_entry = short_rejection and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals