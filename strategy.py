#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and session filter
# - Uses 1h timeframe for precise entry timing around Camarilla H3/L3 levels from prior 4h bar
# - 4h volume spike (>1.5x 20-period average) confirms institutional participation
# - Session filter (08-20 UTC) avoids low-liquidity Asian session noise
# - Discrete position sizing (±0.20) limits drawdown and reduces fee churn
# - Works in bull markets (breakouts with volume) and bear markets (breakdowns with volume)
# - Target: 15-37 trades/year (60-150 total over 4 years) to stay within fee drag limits
# - Camarilla pivots provide mathematically derived support/resistance that adapts to volatility

name = "1h_4h_camarilla_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session hours once (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 4h data ONCE before loop for Camarilla pivots and volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return signals
    
    # Pre-compute 4h indicators
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 4h volume SMA (20-period) for confirmation
    volume_series_4h = pd.Series(volume_4h)
    volume_sma_20_4h = volume_series_4h.rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_4h, volume_sma_20_4h)
    
    # Calculate Camarilla levels for each 4h bar using prior 4h bar's OHLC
    camarilla_h3 = np.full_like(close_4h, np.nan)
    camarilla_l3 = np.full_like(close_4h, np.nan)
    
    for i in range(1, len(df_4h)):
        # Use prior completed 4h bar (i-1) to avoid look-ahead
        high_prev = high_4h[i-1]
        low_prev = low_4h[i-1]
        close_prev = close_4h[i-1]
        range_prev = high_prev - low_prev
        
        if range_prev > 0:
            camarilla_h3[i] = close_prev + 1.1 * range_prev / 6
            camarilla_l3[i] = close_prev - 1.1 * range_prev / 6
    
    # Align Camarilla levels to 1h timeframe (will use previous 4h bar's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if outside trading session or missing data
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Breakout conditions using Camarilla H3/L3 levels
        breakout_long = price_close > camarilla_h3_aligned[i]
        breakout_short = price_close < camarilla_l3_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Camarilla H3 breakout + volume confirmation
        if breakout_long and vol_confirm:
            enter_long = True
        
        # Short: Camarilla L3 breakdown + volume confirmation
        if breakout_short and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Camarilla level break
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below L3 level
            exit_long = price_close < camarilla_l3_aligned[i]
        elif position == -1:
            # Exit short if price breaks above H3 level
            exit_short = price_close > camarilla_h3_aligned[i]
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.20
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals