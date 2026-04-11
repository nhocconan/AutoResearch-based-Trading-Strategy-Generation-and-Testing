#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 12h volume regime + 1d trend filter
# - Long when price breaks above Donchian(20) high + 12h volume > 1.5x 20-period average + price > 1d EMA50
# - Short when price breaks below Donchian(20) low + 12h volume > 1.5x 20-period average + price < 1d EMA50
# - Exit when price crosses opposite Donchian band or volume drops below average
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits for 4h
# - Works in bull markets (breakouts with volume) and bear markets (breakdowns with volume)
# - 1d EMA50 provides trend filter, reducing false signals in choppy markets
# - Volume regime filter ensures trades occur during high conviction moves

name = "4h_12h_1d_donchian_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 12h data ONCE before loop for volume regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return signals
    
    # Pre-compute 12h volume SMA (20-period)
    volume_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA50
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute Donchian channels (20-period) on 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    for i in range(20, n):  # Start after 20-bar warmup for Donchian
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_sma_20_12h_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_12h_current = volume_sma_20_12h_aligned[i]  # Use aligned 12h volume average
        
        # Volume regime: current 12h volume average > 1.5x its 20-period average
        vol_regime = volume_12h_current > 1.5 * volume_sma_20_12h_aligned[i]
        
        # Trend filter
        price_above_ema50 = price_close > ema50_1d_aligned[i]
        price_below_ema50 = price_close < ema50_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = price_high > donchian_high[i]
        breakout_down = price_low < donchian_low[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Donchian breakout up + volume regime + price above 1d EMA50
        if breakout_up and vol_regime and price_above_ema50:
            enter_long = True
        
        # Short: Donchian breakout down + volume regime + price below 1d EMA50
        if breakout_down and vol_regime and price_below_ema50:
            enter_short = True
        
        # Exit conditions: opposite breakout or volume regime fails
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if breakdown OR volume regime fails
            exit_long = breakout_down or (not vol_regime)
        elif position == -1:
            # Exit short if breakout up OR volume regime fails
            exit_short = breakout_up or (not vol_regime)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals