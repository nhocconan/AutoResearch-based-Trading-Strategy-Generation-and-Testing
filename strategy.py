#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 (1h) AND 4h EMA(21) > EMA(50) (bullish trend) AND 1h volume > 2.0x 20-bar avg
# - Short when price breaks below Camarilla L3 (1h) AND 4h EMA(21) < EMA(50) (bearish trend) AND 1h volume > 2.0x 20-bar avg
# - Exit when price returns to Camarilla pivot point (mean reversion to equilibrium)
# - Uses discrete position sizing (0.20) to balance return and drawdown
# - Session filter: 08-20 UTC to avoid low liquidity sessions
# - Target: 60-150 total trades over 4 years = 15-37/year for 1h
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
    
    # Pre-compute 4h EMA trend filter: EMA(21) vs EMA(50)
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_bullish_4h = ema_21_4h > ema_50_4h
    ema_bearish_4h = ema_21_4h < ema_50_4h
    
    # Align HTF indicators to 1h timeframe
    ema_bullish_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_bullish_4h)
    ema_bearish_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_bearish_4h)
    
    # Session filter: 08-20 UTC (avoid low liquidity Asian session)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_4h_aligned[i]) or np.isnan(ema_bearish_4h_aligned[i])):
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
        
        # Calculate 1h Camarilla pivot levels (using last completed 1h bar)
        if i >= 2:  # Need at least 2 bars for calculation
            high_1h = prices['high'].iloc[i-1]
            low_1h = prices['low'].iloc[i-1]
            close_1h = prices['close'].iloc[i-1]
            rang = high_1h - low_1h
            camarilla_pivot = (high_1h + low_1h + close_1h) / 3
            camarilla_h3 = camarilla_pivot + (rang * 1.1 / 4)
            camarilla_l3 = camarilla_pivot - (rang * 1.1 / 4)
        else:
            # Not enough data yet
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Calculate 1h volume confirmation: > 2.0x 20-bar average
        if i >= 20:
            volume_ma_20 = prices['volume'].iloc[i-20:i].mean()
            vol_spike = prices['volume'].iloc[i-1] > (2.0 * volume_ma_20)
        else:
            vol_spike = False
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND 4h bullish trend AND volume spike
            if (prices['high'].iloc[i-1] > camarilla_h3 and 
                ema_bullish_4h_aligned[i] and 
                vol_spike):
                position = 1
                signals[i] = 0.20
            # Short when price breaks below L3 AND 4h bearish trend AND volume spike
            elif (prices['low'].iloc[i-1] < camarilla_l3 and 
                  ema_bearish_4h_aligned[i] and 
                  vol_spike):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot point (mean reversion)
            # Exit when price returns to Camarilla pivot point
            exit_long = position == 1 and prices['low'].iloc[i-1] <= camarilla_pivot
            exit_short = position == -1 and prices['high'].iloc[i-1] >= camarilla_pivot
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals