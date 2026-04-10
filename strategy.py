#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 (12h) AND 12h EMA(50) > EMA(200) (bullish trend) AND 12h volume > 1.5x 20-bar avg
# - Short when price breaks below Camarilla L3 (12h) AND 12h EMA(50) < EMA(200) (bearish trend) AND 12h volume > 1.5x 20-bar avg
# - Exit when price returns to Camarilla pivot point (mean reversion to equilibrium)
# - Uses discrete position sizing (0.25) to balance return and drawdown
# - Camarilla levels provide intraday support/resistance based on prior day's range
# - 12h EMA filter ensures alignment with higher timeframe trend to avoid counter-trend trades
# - Volume confirmation avoids low-liquidity false signals
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: breakouts in trends, mean reversion in ranges

name = "6h_12h_camarilla_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h EMA trend filter: EMA(50) vs EMA(200)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200_12h = pd.Series(close_12h).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_bullish_12h = ema_50_12h > ema_200_12h
    ema_bearish_12h = ema_50_12h < ema_200_12h
    
    # Pre-compute 12h volume confirmation: > 1.5x 20-period average
    volume_12h = df_12h['volume'].values
    volume_20_avg_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (1.5 * volume_20_avg_12h)
    
    # Pre-compute 12h Camarilla pivot levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla calculations
    rang = high_12h - low_12h
    camarilla_pivot = (high_12h + low_12h + close_12h) / 3
    camarilla_h3 = camarilla_pivot + (rang * 1.1 / 4)
    camarilla_l3 = camarilla_pivot - (rang * 1.1 / 4)
    camarilla_h4 = camarilla_pivot + (rang * 1.1 / 2)
    camarilla_l4 = camarilla_pivot - (rang * 1.1 / 2)
    
    # Align HTF indicators to 6h timeframe
    ema_bullish_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_bullish_12h)
    ema_bearish_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_bearish_12h)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pivot)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    
    # Session filter: 08-20 UTC (avoid low liquidity Asian session)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_12h_aligned[i]) or np.isnan(ema_bearish_12h_aligned[i]) or
            np.isnan(vol_spike_12h_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Apply session filter
        if not in_session[i]:
            # Outside session: flatten position
            position = 0
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND 12h bullish trend AND volume spike
            if (prices['close'].iloc[i] > camarilla_h3_aligned[i] and 
                ema_bullish_12h_aligned[i] and 
                vol_spike_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L3 AND 12h bearish trend AND volume spike
            elif (prices['close'].iloc[i] < camarilla_l3_aligned[i] and 
                  ema_bearish_12h_aligned[i] and 
                  vol_spike_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot point (mean reversion)
            # Exit when price returns to Camarilla pivot point
            exit_long = position == 1 and prices['close'].iloc[i] <= camarilla_pivot_aligned[i]
            exit_short = position == -1 and prices['close'].iloc[i] >= camarilla_pivot_aligned[i]
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals