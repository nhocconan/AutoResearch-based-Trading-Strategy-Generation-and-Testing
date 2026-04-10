#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter (EMA20>EMA50) and volume spike confirmation
# - Long: Price breaks above Donchian(20) upper band + 12h EMA20 > EMA50 (uptrend) + 12h volume > 1.5x 20-period MA
# - Short: Price breaks below Donchian(20) lower band + 12h EMA20 < EMA50 (downtrend) + 12h volume > 1.5x 20-period MA
# - Exit: Price crosses Donchian(20) midline (10-period average of upper/lower) or ATR-based stoploss
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 75-200 total trades over 4 years (19-50/year) - proven winning range for 4h
# - Donchian channels provide objective breakout levels; 12h EMA filter ensures trading with higher timeframe trend
# - Volume spike confirms institutional participation, reducing false signals in ranging markets
# - Works in both bull (breakouts continuation) and bear (breakdowns continuation) markets

name = "4h_12h_donchian_breakout_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Pre-compute 4h OHLCV
    open_4h = prices['open'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # Pre-compute 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Donchian(20) channels for 4h
    highest_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate ATR(14) for 4h stoploss
    tr1 = pd.Series(high_4h - low_4h).values
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1))).values
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1))).values
    tr2[0] = tr1[0]  # First bar: no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h EMA(20) and EMA(50) for trend filter
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h volume moving average (20-period)
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(60, n):  # Start after warmup period (need at least 60 for Donchian20 and EMA50)
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(ema_20_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # Get aligned 12h data for current 4h bar (completed 12h bar)
        ema_20_current = ema_20_aligned[i]
        ema_50_current = ema_50_aligned[i]
        volume_ma_current = volume_ma_aligned[i]
        volume_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)[i]
        
        # Trend condition: EMA(20) > EMA(50) for uptrend, EMA(20) < EMA(50) for downtrend
        uptrend = ema_20_current > ema_50_current
        downtrend = ema_20_current < ema_50_current
        
        # Volume spike condition: current 12h volume > 1.5x 20-period MA
        volume_spike = volume_12h_current > 1.5 * volume_ma_current
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian upper + uptrend + volume spike
            if (close_price > donchian_upper[i] and uptrend and volume_spike):
                position = 1
                entry_price = close_price
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian lower + downtrend + volume spike
            elif (close_price < donchian_lower[i] and downtrend and volume_spike):
                position = -1
                entry_price = close_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # Long position
                # Exit 1: Price crosses below Donchian middle (mean reversion)
                if close_price < donchian_middle[i]:
                    exit_signal = True
                # Exit 2: ATR-based stoploss (2 * ATR below entry)
                elif close_price < entry_price - 2.0 * atr[i]:
                    exit_signal = True
                # Exit 3: Opposite Donchian breakout (break below lower band)
                elif close_price < donchian_lower[i]:
                    exit_signal = True
                    
            else:  # position == -1, Short position
                # Exit 1: Price crosses above Donchian middle (mean reversion)
                if close_price > donchian_middle[i]:
                    exit_signal = True
                # Exit 2: ATR-based stoploss (2 * ATR above entry)
                elif close_price > entry_price + 2.0 * atr[i]:
                    exit_signal = True
                # Exit 3: Opposite Donchian breakout (break above upper band)
                elif close_price > donchian_upper[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals