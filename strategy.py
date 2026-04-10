#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level AND 1d close > 1d EMA50 (bullish trend)
# - Short when price breaks below Camarilla L3 level AND 1d close < 1d EMA50 (bearish trend)
# - Volume confirmation: 12h volume > 1.3x 20-period volume SMA
# - Exit: opposite Camarilla breakout or volume drops below average
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-30 trades/year on 12h timeframe to stay within fee drag limits
# - Camarilla levels provide institutional support/resistance that works in both bull and bear markets

name = "12h_1d_camarilla_pivot_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 12h Camarilla pivot levels (using previous bar's OHLC)
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.125*(high-low)
    #            L3 = close - 1.125*(high-low), L4 = close - 1.5*(high-low)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    camarilla_h3 = prev_close + 1.125 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.125 * (prev_high - prev_low)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d close for trend comparison
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate 12h volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(close_1d_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > 1.3 * volume_sma_20[i]
        
        # Trend filter: 1d close vs 1d EMA50
        trend_bullish = close_1d_aligned[i] > ema_50_1d_aligned[i]
        trend_bearish = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_h3[i]  # Break above H3
        breakout_down = close[i] < camarilla_l3[i]  # Break below L3
        
        # Exit conditions: opposite breakout or loss of volume confirmation
        exit_long = breakout_down or not vol_confirm
        exit_short = breakout_up or not vol_confirm
        
        if position == 0:  # Flat - look for entry
            if breakout_up and trend_bullish and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif breakout_down and trend_bearish and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals