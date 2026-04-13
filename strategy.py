#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian channel breakout with 1w volume confirmation and ATR filter
    # Long when price breaks above 20-period Donchian high + 1w volume > 1.5 * 20-period mean + ATR(14) < 0.03 * close
    # Short when price breaks below 20-period Donchian low + same filters
    # Exit when price returns to Donchian midpoint
    # Uses discrete position sizing (0.25) to balance return and drawdown
    # Target: 50-100 total trades over 4 years (~12-25/year) to avoid excessive fee drag
    # Donchian channels provide robust trend-following structure
    # Weekly volume confirmation ensures breakouts have institutional participation
    # ATR filter avoids breakouts during extreme volatility (false breakouts)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channel (20-period) with min_periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 1w volume mean (20-period) with min_periods
    volume_1w = df_1w['volume'].values
    volume_series = pd.Series(volume_1w)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR (14-period) with min_periods
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(high, low, close, 14)
    
    # Align HTF indicators to 1d timeframe
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(atr_1d[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1w volume > 1.5 * 20-period mean
        volume_1w_current = df_1w['volume'].values
        vol_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w_current)
        volume_confirmation = vol_1w_aligned[i] > 1.5 * vol_ma_aligned[i]
        
        # ATR filter: avoid extreme volatility (ATR < 3% of price)
        atr_filter = atr_1d[i] < 0.03 * close[i]
        
        # Breakout conditions with filters
        bullish_breakout = (close[i] > donchian_high[i] and 
                           volume_confirmation and 
                           atr_filter)
        bearish_breakout = (close[i] < donchian_low[i] and 
                           volume_confirmation and 
                           atr_filter)
        
        # Exit conditions: return to Donchian midpoint
        long_exit = close[i] < donchian_mid[i]
        short_exit = close[i] > donchian_mid[i]
        
        if bullish_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_donchian_breakout_1w_volume_atr_filter_v1"
timeframe = "1d"
leverage = 1.0