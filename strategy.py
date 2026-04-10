#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 (4h) AND 4h EMA(50) > EMA(200) (bullish trend) AND 1h volume > 2.0x 24-bar avg
# - Short when price breaks below Camarilla L3 (4h) AND 4h EMA(50) < EMA(200) (bearish trend) AND 1h volume > 2.0x 24-bar avg
# - Exit when price returns to Camarilla pivot point (mean reversion to equilibrium)
# - Uses discrete position sizing (0.20) to minimize fee impact
# - Session filter: 08-20 UTC to avoid low liquidity periods
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Works in both bull and bear markets: breakouts in trends, mean reversion in ranges

name = "1h_4h_camarilla_breakout_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h EMA trend filter: EMA(50) vs EMA(200)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_bullish_4h = ema_50_4h > ema_200_4h
    ema_bearish_4h = ema_50_4h < ema_200_4h
    
    # Pre-compute 4h Camarilla pivot levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla calculations
    rang = high_4h - low_4h
    camarilla_pivot = (high_4h + low_4h + close_4h) / 3
    camarilla_h3 = camarilla_pivot + (rang * 1.1 / 4)
    camarilla_l3 = camarilla_pivot - (rang * 1.1 / 4)
    camarilla_h4 = camarilla_pivot + (rang * 1.1 / 2)
    camarilla_l4 = camarilla_pivot - (rang * 1.1 / 2)
    
    # Align HTF indicators to 1h timeframe
    ema_bullish_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_bullish_4h)
    ema_bearish_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_bearish_4h)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4)
    
    # Pre-compute 1h volume confirmation: > 2.0x 24-period average (24h = 1 day)
    volume_1h = prices['volume'].values
    volume_24_avg_1h = pd.Series(volume_1h).rolling(window=24, min_periods=24).mean().values
    vol_spike_1h = volume_1h > (2.0 * volume_24_avg_1h)
    
    # Session filter: 08-20 UTC (avoid low liquidity Asian session)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_4h_aligned[i]) or np.isnan(ema_bearish_4h_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(vol_spike_1h[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Apply session filter
        if not in_session[i]:
            # Outside session: flatten position
            position = 0
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND 4h bullish trend AND volume spike
            if (prices['close'].iloc[i] > camarilla_h3_aligned[i] and 
                ema_bullish_4h_aligned[i] and 
                vol_spike_1h[i]):
                position = 1
                signals[i] = 0.20
            # Short when price breaks below L3 AND 4h bearish trend AND volume spike
            elif (prices['close'].iloc[i] < camarilla_l3_aligned[i] and 
                  ema_bearish_4h_aligned[i] and 
                  vol_spike_1h[i]):
                position = -1
                signals[i] = -0.20
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
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals