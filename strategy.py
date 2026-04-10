#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level AND 12h EMA50 > EMA200 (bullish trend)
# - Short when price breaks below Camarilla L3 level AND 12h EMA50 < EMA200 (bearish trend)
# - Volume confirmation: 4h volume > 1.3x 20-period volume SMA
# - Exit: opposite Camarilla breakout or trend reversal
# - Position sizing: 0.25 discrete level
# - Target: 20-50 trades/year on 4h timeframe to minimize fee drag
# - Works in bull via breakout continuation, in bear via mean reversion at extreme pivot levels

name = "4h_12h_camarilla_pivot_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Camarilla pivot levels from previous day
    # Using typical pivot: (H+L+C)/3
    # Resistance: H4 = C + (H-L)*1.1/2, H3 = C + (H-L)*1.1/4
    # Support: L4 = C - (H-L)*1.1/2, L3 = C - (H-L)*1.1/4
    typical_price = (high + low + close) / 3
    typical_price_prev = np.roll(typical_price, 1)
    typical_price_prev[0] = np.nan
    
    high_prev = np.roll(high, 1)
    high_prev[0] = np.nan
    low_prev = np.roll(low, 1)
    low_prev[0] = np.nan
    
    # Calculate pivot and ranges
    pivot = (high_prev + low_prev + typical_price_prev) / 3
    range_prev = high_prev - low_prev
    
    camarilla_h3 = pivot + range_prev * 1.1 / 4
    camarilla_l3 = pivot - range_prev * 1.1 / 4
    camarilla_h4 = pivot + range_prev * 1.1 / 2
    camarilla_l4 = pivot - range_prev * 1.1 / 2
    
    # Calculate 12h EMA50 and EMA200 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Calculate 4h volume SMA for regime filter
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_200_12h_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 4h volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > 1.3 * volume_sma_20[i]
        
        # Trend filter: 12h EMA50 vs EMA200
        trend_bullish = ema_50_12h_aligned[i] > ema_200_12h_aligned[i]
        trend_bearish = ema_50_12h_aligned[i] < ema_200_12h_aligned[i]
        
        # Camarilla breakout signals (using previous bar's levels to avoid look-ahead)
        breakout_up = close[i] > camarilla_h3[i-1]  # Break above H3
        breakout_down = close[i] < camarilla_l3[i-1]  # Break below L3
        
        # Exit conditions: opposite breakout or trend reversal
        exit_long = breakout_down or not trend_bullish
        exit_short = breakout_up or not trend_bearish
        
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